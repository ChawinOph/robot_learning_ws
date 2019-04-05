from math import sin, cos
import ast
import math
import numpy as np
from scipy.integrate import solve_ivp
from simulated_dynamics import NumpySimulatedDynamics
from friction import Friction


def xaxis():
    x = np.zeros((2,1))
    x[0] = 1
    return x


def yaxis():
    y = np.zeros((2,1))
    y[1] = 1
    return y


def rot(theta):
    R = np.zeros((2,2))
    R[0,0] = math.cos(theta)
    R[0,1] = -math.sin(theta)
    R[1,0] = math.sin(theta)
    R[1,1] = math.cos(theta)
    return R


def wrap(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi


class ChainNumpySimulatedDynamics(NumpySimulatedDynamics):
    def __init__(self, delta_t=None, num_links=None, link_mass=None, link_length=None,
                 torques_max=None, friction=None, gravity_y=None, gravity_z=None,
                 solver_type=None, wrap_angles=None):
        NumpySimulatedDynamics.__init__(self, delta_t)
        self.num_links = num_links
        self.link_mass = link_mass
        self.link_length = link_length
        self.torques_max = torques_max

        self.friction = friction

        self.gravity_y = gravity_y
        self.gravity_z = gravity_z
        self.solver_type = solver_type
        self.wrap_angles = wrap_angles
        self._solve_time = 0.0
        self.energy = 0.0
        self.joint_limits = False

        # Generalized forces not affected by actions
        self.fx = 0.0
        self.fy = 0.0
        self.t0 = 0.0

        self.action = None

    def initialize_from_config(self, config_data, section_name):
        NumpySimulatedDynamics.initialize_from_config(self, config_data, section_name)
        self.num_links = config_data.getint(section_name, 'num_links')
        self.link_length = config_data.getfloat(section_name, 'link_length')
        self.link_mass = config_data.getfloat(section_name, 'link_mass')

        self.torques_max = config_data.getfloat(section_name, 'torques_max')
        friction_section_name = config_data.get(section_name, 'friction')
        self.friction = Friction()
        self.friction.initialize_from_config(config_data, friction_section_name)
        self.gravity_y = config_data.getfloat(section_name, 'gravity_y')
        self.gravity_z = config_data.getfloat(section_name, 'gravity_z')
        self.solver_type = config_data.get(section_name, 'solver_type')

        if config_data.has_option(section_name, 'wrap_angles'):
            self.wrap_angles = config_data.getboolean(section_name, 'wrap_angles')
        else:
            self.wrap_angles = False

        if config_data.has_option(section_name, 'joint_limits'):
            self.joint_limits = config_data.getboolean(section_name, 'joint_limits')
            if self.joint_limits:
                self.qmax = config_data.getfloat(section_name, 'qmax')
        else:
            self.joint_limits = False
        self.energy = 0

    def compute_energy(self, state):
        raise NotImplementedError('compute_energy has to be different for lagrangian and newtonian chains')
        # num_links = self.get_state_dim()/2
        # q = state[0:num_links]
        # qd = state[num_links:]
        # pos_y = np.zeros(num_links)
        #
        # m = self.link_mass
        # l = self.link_length
        # mI = m*l*l/12
        # g = self.gravity_y
        #
        # pos_y[0] = 0.5*l*sin(q[0])
        # for i in range(1, num_links):
        #     pos_y[i] = pos_y[i-1] + 0.5*l*(sin(q[i-1]) + sin(q[i]))
        #
        # vel = np.zeros((num_links, 2))
        # vel[0, 0] = -0.5 * l * sin(q[0]) * qd[0]
        # vel[0, 1] = 0.5 * l * cos(q[0]) * qd[0]
        # for i in range(1, num_links):
        #     vel[i, 0] = vel[i - 1, 0] - 0.5 * l * (sin(q[i - 1]) * qd[i - 1] + sin(q[i]) * qd[i])
        #     vel[i, 1] = vel[i - 1, 1] + 0.5 * l * (cos(q[i - 1]) * qd[i - 1] + cos(q[i]) * qd[i])
        #
        # T = 0.0
        # V = 0.0
        # for i in range(num_links):
        #     T += 0.5 * m * (vel[i, 0] ** 2 + vel[i, 1] ** 2)
        #     T += 0.5 * mI * qd[i] ** 2
        #     V += m * g * pos_y[i]
        #
        # self.energy = [T, V]

    def advance(self, state, actions, delta_t=None):
        if delta_t is None: delta_t = self.delta_t
        state_dim = self.get_state_dim()
        assert (state.shape == (self.get_state_dim(), 1))
        assert (actions.shape[0] == self.get_action_dim())
        # approximate integration with Euler's method. Alternative is using
        # more accurate scipy initial value problem (IVP) solver with RK4
        # and adaptive step-size
        if self.solver_type == 'euler':
            new_states = self.advance_euler(state, actions, delta_t)
        elif self.solver_type == 'ivp':
            new_states = self.advance_ivp(state, actions, delta_t)
        else:
            raise ValueError('solver type: {} not recognized'.format(self.solver_type))

        if self.wrap_angles:
            new_states = self.wrap(new_states.reshape(-1, 1))

        if self.joint_limits:
            new_states = self.apply_joint_limits(new_states.reshape(-1, 1))

        # self.compute_energy(new_states)
        return new_states.reshape((state.shape[0], actions.shape[1]))

    def wrap(self, state):
        raise NotImplementedError

    def apply_joint_limits(self, state):
        raise NotImplementedError

    def _integrate_euler(self, state, action, delta_t):
        raise NotImplementedError

    def advance_euler(self, state, actions, delta_t):
        new_states = []
        for i in range(actions.shape[1]):
            new_state = self._integrate_euler(state, actions[:, i], delta_t)
            new_states.append(new_state)
        return np.vstack(new_states).T

    def advance_ivp(self, state, actions, delta_t):
        new_states = []
        for i in range(actions.shape[1]):
            self.action = actions[:, i]
            new_state = solve_ivp(self.dynamics, [0, delta_t], state.reshape(-1, ), t_eval=[delta_t]).y
            self.action = None
            new_states.append(new_state)
        return np.vstack(new_states).T

    def get_num_links(self):
        return self.num_links

    def get_link_length(self):
        return self.link_length

    def get_link_lengths(self):
        return np.full((self.num_links,), self.link_length)

    def get_state_dim(self):
        raise NotImplementedError()

    def get_action_dim(self):
        raise NotImplementedError()

    def dynamics(self, t, y):
        raise NotImplementedError

    def get_chain_state_from_robot_state(self, state):
        raise NotImplementedError

    def get_robot_state_from_chain_state(self, state):
        raise NotImplementedError

    def get_chain_action_from_robot_action(self, action):
        raise NotImplementedError

    def get_pos_0(self, state):
        raise NotImplementedError

    def get_vel_0(self, state):
        raise NotImplementedError

    def get_q(self, state):
        raise NotImplementedError

    def get_qd(self, state):
        raise NotImplementedError


class ArmNumpySimulatedDynamics(ChainNumpySimulatedDynamics):
    def __init__(self, delta_t=None, num_links=None, link_mass=None, link_length=None, torques_max=None, friction=None,
                 gravity_y=None, gravity_z=None, solver_type=None, wrap_angles=None):
        ChainNumpySimulatedDynamics.__init__(self, delta_t, num_links, link_mass, link_length, torques_max, friction, gravity_y,
                                             gravity_z, solver_type, wrap_angles)

    def get_state_dim(self):
        return 2 * self.num_links

    def get_action_dim(self):
        return self.num_links

    def get_chain_state_from_robot_state(self, state):
        state = np.vstack([np.zeros((2, 1)), state[:self.num_links].reshape(-1, 1),
                           np.zeros((2, 1)), state[self.num_links:].reshape(-1, 1)])
        return state

    def get_robot_state_from_chain_state(self, state):
        q_idxs = np.arange(2, self.num_links+2)
        qd_idxs = np.arange(self.num_links+4, 2*self.num_links+4)
        robot_idxs = np.concatenate([q_idxs, qd_idxs])
        return state[robot_idxs, :]

    def get_chain_action_from_robot_action(self, action):
        return action

    def _integrate_euler(self, state, action, delta_t):
        self.action = action
        dydt = self.dynamics(0, state.reshape(-1, ))
        new_state = state.reshape(-1, ) + dydt * delta_t
        new_state[:self.num_links] += 0.5 * dydt[self.num_links:] * delta_t ** 2
        self.action = None
        return new_state

    def dynamics(self, t, y):
        raise NotImplementedError

    def get_pos_0(self, state):
        return np.array([[0.0], [0.0]])

    def get_vel_0(self, state):
        return np.array([[0.0], [0.0]])

    def get_q(self, state):
        return state[:self.num_links]

    def get_qd(self, state):
        return state[self.num_links:]

    def delta_state(self, state1, state2):
        """ returns state1 - state2 """
        delta_q = wrap(self.get_q(state1) - self.get_q(state2))
        delta_qd = self.get_qd(state1) - self.get_qd(state2)
        delta_state = np.vstack([delta_q, delta_qd])
        return delta_state

    def wrap(self, state):
        q = wrap(self.get_q(state))
        qd = self.get_qd(state)
        state = np.vstack((q, qd))
        return state

    def apply_joint_limits(self, state):
        raise NotImplementedError
        q = self.get_q(state)
        qd = self.get_qd(state)
        qmax = self.qmax
        for i in range(self.num_links):
            if q[i] > qmax:
                q[i] = qmax
                qd[i] = 0.0
            elif q[i] < -qmax:
                q[i] = -qmax
                qd[i] = 0.0
        state = np.vstack((q, qd))
        return state
