#!/usr/bin/env python

import numpy as np
import rospy
import threading
import time

from newtonian_chain_numpy_simulated_dynamics import NewtonianArmDynamics
from simulated_robot import SimulatedRobot
from friction import Friction
from simulated_chain_robot_gui import SimulatedChainRobotGUI

from robot_sim.msg import RobotState

class RobotGUI(object):
    def __init__(self):
        self.robot_names=['real_robot','fake_robot']
        self.dynamics = NewtonianArmDynamics(delta_t=1,
                                             num_links=3,
                                             link_mass=10,
                                             link_length=0.5,
                                             torques_max=100.0,
                                             friction=Friction(apply_viscous_joint_friction=False,
                                                               mu_viscous_joint=0.0,
                                                               apply_dry_friction=True,
                                                               mu_dry_forward=1.0,
                                                               mu_dry_sideways=0.4,
                                                               apply_viscous_friction=False,
                                                               mu_viscous_forward=0.0,
                                                               mu_viscous_sideways=0.0),
                                             gravity_y=-1,
                                             gravity_z=0,
                                             solver_type='euler',
                                             wrap_angles=True,
                                             residue_limit=0.0001)

        self.gui = SimulatedChainRobotGUI(render_rate=0.01,
                                          record_fps=None,
                                          record_sim=False,
                                          time_frame='robot',
                                          lim_x=[-2,2],
                                          lim_y=[-2,2],
                                          subject=None)        
        
        self.robots=[]
        initial_state = [-1.57, 0.0, 0.0, 0.0, 0.0, 0.0]
        for name in self.robot_names:
            robot = SimulatedRobot(dynamics=self.dynamics,
                                   data_names=None,
                                   steps_per_action=1)
            robot.set_state(np.asarray(initial_state).reshape((-1, 1)))
            self.robots.append(robot)
            self.gui.add_subject(robot)

        self.sub = rospy.Subscriber(name = "/robot_states", data_class = RobotState,
                                    callback = self.callback, queue_size = 100)

    def callback(self, msg):
        robot = None
        for i in range(0,len(self.robot_names)):
            if self.robot_names[i] == msg.robot_name: robot = self.robots[i]
        if robot == None:
            print "Warning: message received for unknown robot"
            return
        robot.set_state(np.asarray(msg.robot_state))

    def update(self):
        self.gui.render()
    
def my_thread():
    rospy.spin()
            
if __name__ == '__main__':
    rospy.init_node('robot_gui', anonymous=True)
    gui = RobotGUI()

    my_thread = threading.Thread(target=my_thread)
    my_thread.daemon = True
    my_thread.start()

    while not rospy.is_shutdown():
        gui.update()
        time.sleep(0.005)
    
