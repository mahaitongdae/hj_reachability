import jax.numpy as jnp

from hj_reachability import dynamics
from hj_reachability import sets

class DoubleInt(dynamics.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 range=5.,
                 max_u = 0.5,
                 control_mode="min",
                 disturbance_mode="max",
                 control_space=None,
                 disturbance_space=None):
        if control_space is None:
            control_space = sets.Box(lo=jnp.array([-max_u]), hi=jnp.array([max_u]))
        if disturbance_space is None:
            disturbance_space = sets.Box(lo=jnp.array([-max_u]), hi=jnp.array([max_u]))
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        x_1, x_2 = state
        # v_a, v_b = self.evader_speed, self.pursuer_speed
        return jnp.array([x_2, 0])

    def control_jacobian(self, state, time):
        x, y, _ = state
        return jnp.array([
            [0],
            [1],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [0.],
            [0.],
        ])