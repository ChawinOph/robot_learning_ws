class Friction(object):
    def __init__(self, apply_viscous_joint_friction=False, mu_viscous_joint=0.0, apply_dry_friction=False,
                 mu_dry_forward=0.0, mu_dry_sideways=0.0, apply_viscous_friction=False, mu_viscous_forward=0.0,
                 mu_viscous_sideways=0.0):
        self.apply_viscous_friction = apply_viscous_friction
        self.mu_viscous_joint = mu_viscous_joint
        self.apply_dry_friction = apply_dry_friction
        self.mu_dry_forward = mu_dry_forward
        self.mu_dry_sideways = mu_dry_sideways
        self.apply_viscous_friction = apply_viscous_friction
        self.mu_viscous_forward = mu_viscous_forward
        self.mu_viscous_sideways = mu_viscous_sideways
        self.apply_viscous_joint_friction = apply_viscous_joint_friction
        self.mu_viscous_joint = mu_viscous_joint

    def initialize_from_config(self, config_data, section_name):
        self.apply_viscous_joint_friction = config_data.getboolean(section_name, 'apply_viscous_joint_friction')
        if self.apply_viscous_joint_friction:
            self.mu_viscous_joint = config_data.getfloat(section_name, 'mu_viscous_joint')
        self.apply_viscous_friction = config_data.getboolean(section_name, 'apply_viscous_friction')
        if self.apply_viscous_friction:
            self.mu_viscous_forward = config_data.getfloat(section_name, 'mu_viscous_forward')
            self.mu_viscous_sideways = config_data.getfloat(section_name, 'mu_viscous_sideways')
        self.apply_dry_friction = config_data.getboolean(section_name, 'apply_dry_friction')
        if self.apply_dry_friction:
            self.mu_dry_forward = config_data.getfloat(section_name, 'mu_dry_forward')
            self.mu_dry_sideways = config_data.getfloat(section_name, 'mu_dry_sideways')