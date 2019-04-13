#!/usr/bin/env python

# ROS environment
import time
import rospy 
from robot_sim.srv import RobotAction
from robot_sim.srv import RobotActionRequest
from robot_sim.srv import RobotActionResponse
from robot_sim.msg import RobotState

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import numpy as np
import math

class FakeRobot(object):
	def __init__(self):
		# Wait unitil the real_robot node is available
		rospy.wait_for_service('real_robot')
		# Create the service caller from real_robot
		self.real_robot_action = rospy.ServiceProxy('real_robot', RobotAction)

		self.pub = rospy.Publisher("/robot_states", RobotState, queue_size=100)

		# obtain training data set from real_robot
		print "Obtaining training data from real robot within 10 sec"

		req_real = RobotActionRequest()

		# apply a constant effort as long as the testing code (or should it be longer)
		perturb_steps = 200
		# apply constant torque # scale torque values like in executive.py
		torque = np.random.rand(1,3)
		torque[0,0] = (2 * torque[0,0] - 1.0) * 1.0
		torque[0,1] = (2 * torque[0,1] - 1.0) * 0.5
		torque[0,2] = (2 * torque[0,2] - 1.0) * 0.25
		req_real.action = torque.reshape(3)
		
		for i in range(perturb_steps):
			req_real.reset = False
			resp_real = self.real_robot_action(req_real)
			print resp_real
			# for visualizeing the perturbed real_robot
			message = RobotState()
			message.robot_name=str('real_robot')
			message.robot_state = resp_real.robot_state
			self.pub.publish(message)

		# should visualize the data via gui when we perturb real_robot

		# loop until the real_robot stop poviding service
			# request joint values from given joint effort command
			# reset config if needed?

		# self.training_data = 
		# train data via NN using Pytorch

		# store the trained networked and ready to provide the service
		self.fake_robot_service = rospy.Service('fake_robot', RobotAction, self.fake_robot_action)

	def fake_robot_action(self, req):
		# what to feed the trained NN besides the action? the reset?
		# the robot might have to "remember its own post if thet reset is false"
		# return fake_robot state from input to NN
		pass

if __name__ == '__main__':
	rospy.init_node('fake_robot', anonymous=True)
	fake_robot = FakeRobot();
	print "Finished training fake_robot"
	print "Fake robot now spinning"
	rospy.spin()