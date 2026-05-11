import numpy as np
from scipy.spatial.transform import Rotation as R


class ConstantVelocity:
    def __call__(self, q_prev, t_prev, q_curr, t_curr):
        # Linear velocity
        v = t_curr - t_prev
        t_next = t_curr + v

        # Rotational "velocity"
        r_prev = R.from_quat(q_prev)
        r_curr = R.from_quat(q_curr)
        r_delta = r_prev.inv() * r_curr   # rotation from prev->curr
        r_next = r_curr * r_delta         # predict next rotation

        return r_next.as_quat(), t_next

class IMUMotionModel:
    def __call__(self, R_i, v_i, p_i, preintegration):
        delta_R, delta_v, delta_p = preintegration.get_measurement()
        dt   = preintegration.get_dt()
        g    = preintegration.imu.get_gravity()

        # propagate
        R_j = R_i @ delta_R
        v_j = v_i + g * dt        + R_i @ delta_v
        p_j = p_i + v_i * dt      + 0.5 * g * dt**2  +  R_i @ delta_p

        return R_j, v_j, p_j


class IMUPreIntegration:
    def __init__(self, imu):
        self.imu = imu
        noise = imu.get_noise_params()
        self.sigma_a  = noise['sigma_a']
        self.sigma_g  = noise['sigma_g']
        self.sigma_ba = noise['sigma_ba']
        self.sigma_bg = noise['sigma_bg']
        self.reset()
    

    def reset(self, bias_accel=None, bias_gyro=None):
        self.delta_p = np.zeros(3)          
        self.delta_v = np.zeros(3)         
        self.delta_R = np.eye(3)           
        self.bias_accel = bias_accel if bias_accel is not None else np.zeros(3)
        self.bias_gyro  = bias_gyro  if bias_gyro  is not None else np.zeros(3)
        self.cov = np.zeros((15, 15))

        self.dp_dba = np.zeros((3, 3))  
        self.dp_dbg = np.zeros((3, 3))  
        self.dv_dba = np.zeros((3, 3))   
        self.dv_dbg = np.zeros((3, 3))  
        self.dR_dbg = np.zeros((3, 3))  

        self.dt_sum = 0.0         
        self.prev_timestamp = None

    def integrate(self, t0, t1):
        """Integrate all IMU samples between t0 and t1."""
        samples = self.imu.get_data_range(t0, t1)
        if not samples:
            return

        for i in range(len(samples) - 1):
            s0, s1 = samples[i], samples[i + 1]
            dt = (s1['t'] - s0['t']) * 1e-9   # nanoseconds → seconds
            if dt <= 0:
                continue
            self._integrate_step(s0, s1, dt)

    def _integrate_step(self, s0, s1, dt):

        a0 = np.array(s0['accel']) - self.bias_accel
        a1 = np.array(s1['accel']) - self.bias_accel
        g0 = np.array(s0['gyro'])  - self.bias_gyro
        g1 = np.array(s1['gyro'])  - self.bias_gyro

        a_mid = 0.5 * (a0 + a1)   
        g_mid = 0.5 * (g0 + g1)   
        dR_step = R.from_rotvec(g_mid * dt).as_matrix()
        self.delta_p += self.delta_v * dt + 0.5 * (self.delta_R @ a_mid) * dt**2
        self.delta_v += self.delta_R @ a_mid * dt
        self.delta_R  = self.delta_R @ dR_step

        self._propagate_covariance(a_mid, g_mid, dR_step, dt)
        self._update_jacobians(a_mid, dR_step, dt)

        self.dt_sum += dt



    def _propagate_covariance(self, a_mid, g_mid, dR_step, dt):
   
        Ra = self.delta_R                   
        a_skew = _skew(Ra @ a_mid)

        # state transition matrix F  (15×15)
        F = np.eye(15)
        F[0:3,  3:6]  =  np.eye(3) * dt
        F[0:3,  6:9]  = -0.5 * a_skew * dt**2
        F[0:3,  9:12] = -0.5 * Ra * dt**2
        F[3:6,  6:9]  = -a_skew * dt
        F[3:6,  9:12] = -Ra * dt
        F[6:9,  6:9]  =  dR_step.T
        F[6:9, 12:15] = -np.eye(3) * dt

        # noise input matrix G  (15×12)
        G = np.zeros((15, 12))
        G[3:6,  0:3]  = Ra * dt
        G[6:9,  3:6]  = np.eye(3) * dt
        G[9:12, 6:9]  = np.eye(3) * dt
        G[12:15,9:12] = np.eye(3) * dt

        # discrete noise covariance Q  (12×12)
        sa2  = (self.sigma_a  ** 2) / dt
        sg2  = (self.sigma_g  ** 2) / dt
        sba2 = (self.sigma_ba ** 2) * dt
        sbg2 = (self.sigma_bg ** 2) * dt
        Q = np.diag([sa2]*3 + [sg2]*3 + [sba2]*3 + [sbg2]*3)

        self.cov = F @ self.cov @ F.T + G @ Q @ G.T


    def _update_jacobians(self, a_mid, dR_step, dt):
        Ra = self.delta_R
        a_skew = _skew(Ra @ a_mid)

        self.dp_dba += self.dv_dba * dt - 0.5 * Ra * dt**2
        self.dp_dbg += self.dv_dbg * dt - 0.5 * a_skew @ self.dR_dbg * dt**2

        self.dv_dba += -Ra * dt
        self.dv_dbg += -a_skew @ self.dR_dbg * dt

        self.dR_dbg  = dR_step.T @ self.dR_dbg - np.eye(3) * dt


    def correct_for_bias_update(self, delta_ba, delta_bg):
        """
        Apply first-order correction when optimiser updates biases.
        Call this instead of re-integrating from scratch.
        """
        dR_corr = R.from_rotvec(self.dR_dbg @ delta_bg).as_matrix()
        self.delta_R  = self.delta_R @ dR_corr
        self.delta_v += self.dv_dba @ delta_ba + self.dv_dbg @ delta_bg
        self.delta_p += self.dp_dba @ delta_ba + self.dp_dbg @ delta_bg


    def get_measurement(self):
        """Return preintegrated (ΔR, Δv, Δp) for use as a factor."""
        return self.delta_R.copy(), self.delta_v.copy(), self.delta_p.copy()

    def get_covariance(self):
        return self.cov.copy()

    def get_dt(self):
        return self.dt_sum

def _skew(v):
    """3-vector → 3×3 skew-symmetric matrix."""
    return np.array([
        [ 0,    -v[2],  v[1]],
        [ v[2],  0,    -v[0]],
        [-v[1],  v[0],  0   ],
    ])