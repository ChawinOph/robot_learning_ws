from __future__ import absolute_import, division, print_function, unicode_literals
import ast
import numpy as np
import copy
from robot import Robot


class SimulatedRobot(Robot):
    def __init__(self, dynamics=None, data_names=None, steps_per_action=None):
        Robot.__init__(self)
        self.dynamics = dynamics
        self.steps_per_action = steps_per_action
        self.data_names = data_names
        self.steps = 0
        self.state = None

    def initialize_from_config(self, config_data, section_name):
        Robot.initialize_from_config(self, config_data, section_name)
        robot_section_name = config_data.get(section_name, 'robot')
        self.steps_per_action = config_data.getint(robot_section_name, 'steps_per_action')
        dynamics_section_name = config_data.get(robot_section_name, 'dynamics')
        self.robot_type = config_data.get(dynamics_section_name, 'type')
        self.dynamics = factory_from_config(simulated_dynamics_factory_base, config_data, dynamics_section_name)
        self.data_names = factory_from_config(data_names_factory_base, config_data, dynamics_section_name)

    def set_state_to_zero(self):
        state = np.zeros((self.get_state_dim(), 1))
        self.set_state(state)

    def get_time(self):
        return self.steps * self.dynamics.get_delta_t()

    def get_state(self):
        if self.state is None:
            raise ValueError('ROBOT STATE HAS NOT BEEN SET. Make sure you are setting the state. It is not initialized to anything (to avoid ambiguities), you must have a line in code that calls set_state().')
        timestamp = np.asarray([self.get_time()])
        return copy.copy(self.state), timestamp

    def set_state(self, x):
        self.state = copy.copy(x)

    def take_action(self, action):
        for _ in range(self.steps_per_action):
            self.set_state(self.dynamics.advance(self.state, action))
            self.steps += 1

    def get_state_dim(self):
        return self.dynamics.get_state_dim()

    def get_action_dim(self):
        return self.dynamics.get_action_dim()

    def reset_time(self):
        self.steps = 0
