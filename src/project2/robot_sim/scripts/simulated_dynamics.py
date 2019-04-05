#!/usr/bin/env python
import numpy as np

class SimulatedDynamics(object):
    def __init__(self, delta_t=None):
        self.dynamics_type = None
        self.residue_limit_flag = False
        self.delta_t = delta_t

    def initialize_from_config(self, config_data, section_name):
        self.dynamics_type = config_data.get(section_name, 'type')
        self.delta_t = float(config_data.get(section_name, 'delta_t'))

    def get_data_type(self):
        return self.data_type

    def get_delta_t(self):
        return self.delta_t

    def get_state_dim(self):
        raise NotImplementedError

    def get_action_dim(self):
        raise NotImplementedError

    def advance(self, state, actions, delta_t=None):
        raise NotImplementedError

    def delta_state(self, state1, state2):
        return state1 - state2


class NumpySimulatedDynamics(SimulatedDynamics):
    def __init__(self, delta_t=None):
        SimulatedDynamics.__init__(self, delta_t)
        self.delta_t = delta_t
        self.data_type = np.ndarray

    def initialize_from_config(self, config_data, section_name):
        SimulatedDynamics.initialize_from_config(self, config_data, section_name)

    def get_state_dim(self):
        raise NotImplementedError

    def get_action_dim(self):
        raise NotImplementedError

    def advance(self, state, actions, delta_t=None):
        raise NotImplementedError


