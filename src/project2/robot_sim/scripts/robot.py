from __future__ import absolute_import, division, print_function, unicode_literals

class Robot(object):
    def __init__(self):
        pass

    def initialize(self):
        pass

    def initialize_from_config(self, config_data, section_name):
        pass

    def get_state(self):
        raise NotImplementedError

    def take_action(self, goal):
        raise NotImplementedError

    def initialize_loggers(self):
        pass


