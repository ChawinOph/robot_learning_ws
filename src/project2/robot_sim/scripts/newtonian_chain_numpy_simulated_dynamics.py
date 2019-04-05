from __future__ import absolute_import, division, print_function, unicode_literals
from chain_numpy_simulated_dynamics import ArmNumpySimulatedDynamics
from newtonian_dynamics import ArmDynamicsWrapper

class NewtonianArmDynamics(ArmNumpySimulatedDynamics):
    def __init__(self, delta_t=None, num_links=None, link_mass=None, link_length=None, torques_max=None, friction=None,
                 gravity_y=None, gravity_z=None, solver_type=None, wrap_angles=None, residue_limit=None):
        ArmNumpySimulatedDynamics.__init__(self, delta_t, num_links, link_mass, link_length, torques_max, friction,
                                           gravity_y, gravity_z, solver_type, wrap_angles)
        self.residue_limit = residue_limit
        self.dynamics_wrapper = ArmDynamicsWrapper(
            self.num_links,
            self.link_mass,
            self.link_length,
            self.torques_max,
            self.friction.mu_viscous_joint,
            self.friction.mu_viscous_forward,
            self.friction.mu_viscous_sideways,
            self.friction.mu_dry_forward,
            self.friction.mu_dry_sideways,
            self.gravity_y,
            self.gravity_z,
        )

    def initialize_from_config(self, config_data, section_name):
        ArmNumpySimulatedDynamics.initialize_from_config(self, config_data, section_name)

        self.dynamics_wrapper = ArmDynamicsWrapper(
            self.num_links,
            self.link_mass,
            self.link_length,
            self.torques_max,
            self.friction.mu_viscous_joint,
            self.friction.mu_viscous_forward,
            self.friction.mu_viscous_sideways,
            self.friction.mu_dry_forward,
            self.friction.mu_dry_sideways,
            self.gravity_y,
            self.gravity_z,
        )

    def dynamics(self, t, y):
            u = self.action
            return self.dynamics_wrapper.compute_xd(y, u).reshape(-1)

    def compute_energy(self, state):
        self.energy = self.dynamics_wrapper.compute_energy(state)


