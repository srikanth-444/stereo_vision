import numpy as np

class IMU:
    def __init__(self, imu_config, interface):
        self.interface = interface

        # sensor noise characteristics (hardware constants)
        self.g_noise_density  = imu_config.get('gyroscope_noise_density')     # σ_g  [rad/s/√Hz]
        self.g_random_walk    = imu_config.get('gyroscope_random_walk')        # σ_bg [rad/s²/√Hz]
        self.a_noise_density  = imu_config.get('accelerometer_noise_density')  # σ_a  [m/s²/√Hz]
        self.a_random_walk    = imu_config.get('accelerometer_random_walk')    # σ_ba [m/s³/√Hz]
        self.rate             = imu_config.get('rate')                         # Hz



    def get_data(self, timestamp):
        """Read one raw sample — no bias correction here."""
        return self.interface.read(timestamp)

    def get_data_range(self, t0, t1):
        """Read a batch of raw samples in [t0, t1]."""
        return self.interface.read_batch(t0, t1)

    def get_dt(self):
        return 1.0 / self.rate

    def is_available(self):
        return self.interface.is_available()

    def get_continuous_noise_cov(self):
        sa2 = self.a_noise_density ** 2
        sg2 = self.g_noise_density ** 2
        return np.diag([sa2, sa2, sa2, sg2, sg2, sg2])

    def get_discrete_noise_cov(self, dt):
        return self.get_continuous_noise_cov() / dt

    def get_bias_noise_cov(self):
        sba2 = self.a_random_walk ** 2
        sbg2 = self.g_random_walk ** 2
        return np.diag([sba2, sba2, sba2, sbg2, sbg2, sbg2])

    def get_noise_params(self):
        return {
            'sigma_a':  self.a_noise_density,
            'sigma_g':  self.g_noise_density,
            'sigma_ba': self.a_random_walk,
            'sigma_bg': self.g_random_walk,
        }




