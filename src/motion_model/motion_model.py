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