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
		# wait unitil the real_robot node is available
		rospy.wait_for_service('real_robot')
		# create the service caller from real_robot
		self.real_robot_action = rospy.ServiceProxy('real_robot', RobotAction)
		self.pub = rospy.Publisher("/robot_states", RobotState, queue_size=100)
		# obtain training data set from real_robot
		print "Obtaining training data from real robot..."
		self.num_tests = 20 
		self.perturb_steps = 200
		features = []
		for i in range(0, self.num_tests):
			action = np.random.rand(1,3)
			action[0,0] = (2 * action[0,0] - 1.0) * 1.0
			action[0,1] = (2 * action[0,1] - 1.0) * 0.5
			action[0,2] = (2 * action[0,2] - 1.0) * 0.25
			self.perturb(action.reshape(3))

			
			time.sleep(1.00)
		# self.training_data = 
		# train data via NN using Pytorch
		print "Training the network..."
		self.network = MyDNN(features.shape[1])
		self.trainer = MyDNNTrain(self.network)
		self.trainer.train(labels, features)
		print "Finished training fake_robot"
		# create service from fake robot
		self.fake_robot_service = rospy.Service('fake_robot', RobotAction, self.fake_robot_action)

	def perturb(self, action):
		req_real = RobotActionRequest()
		req_real.reset = True
		# reset config
		self.real_robot_action(req_real)
		# apply constant action
		for j in range(self.perturb_steps):
			req_real = RobotActionRequest()
			req_real.reset = False
			print action
			req_real.action = action
			resp_real = self.real_robot_action(req_real)
			print resp_real
			# visualizeing the perturbed real_robot in gui
			message = RobotState()
			message.robot_name=str('real_robot')
			message.robot_state = resp_real.robot_state
			self.pub.publish(message)

	def viz_robot(self, robot_name, robot_state):
		pass

	def fake_robot_action(self, req_fake):
		# what to feed the trained NN besides the action? the reset?
		# the robot might have to "remember its own post if thet reset is false"
		# return fake_robot state from input to NN
		resp_fake = RobotActionResponse()
		
		# if the request has the reset as true, reset the current robot config to the 
		predictions = self.network.predict(features)
		return 

class MyDNN(nn.Module):
	def __init__(self, input_dim):
		super(MyDNN, self).__init__()
		self.fc1 = nn.Linear(input_dim, 32)
		self.fc2 = nn.Linear(32, 32)
		self.fc3 = nn.Linear(32, 1)

		def forward(self, x):
			x = F.relu(self.fc1(x))
			x = F.relu(self.fc2(x))
			x = self.fc3(x)
			return x

		def predict(self, features):
			""" Function receives a numpy array, converts to torch, returns numpy again"""
			self.eval()	#Sets network in eval mode (vs training mode)
			features = torch.from_numpy(features).float()
			return self.forward(features).detach().numpy()

class MyDataset(Dataset):
	def __init__(self, labels, features):
		super(MyDataset, self).__init__()
		self.labels = labels
		self.features = features

	def __len__(self):
		return self.features.shape[0]

	def __getitem__(self, idx):		
		#This tells torch how to extract a single datapoint from a dataset, Torch randomized and needs a way to get the nth-datapoint
		feature = self.features[idx]
		label = self.labels[idx]
		return {'feature': feature, 'label': label}

class MyDNNTrain(object):
	def __init__(self, network): #Networks is of datatype MyDNN
		self.network = network
		self.learning_rate = .01 # default: 0.01
		self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.learning_rate)
		self.criterion = nn.MSELoss()
		self.num_epochs = 500 	# default: 500
		self.batchsize = 100 	# default: 100
		self.shuffle = True 	# default: True

	def train(self, labels, features):
		self.network.train()
		dataset = MyDataset(labels, features)
		loader = DataLoader(dataset, shuffle=self.shuffle, batch_size = self.batchsize)
		for epoch in range(self.num_epochs):
			self.train_epoch(loader)

	def train_epoch(self, loader):
		total_loss = 0.0
		for i, data in enumerate(loader):
			features = data['feature'].float()
			labels = data['label'].float()
			self.optimizer.zero_grad()
			predictions = self.network(features)
			loss = self.criterion(predictions, labels)
			loss.backward()
			total_loss += loss.item()
			self.optimizer.step()
		print 'loss', total_loss/i

def main():
	rospy.init_node('fake_robot', anonymous=True)
	fake_robot = FakeRobot();
	print "Fake robot now spinning"
	rospy.spin()

if __name__ == '__main__':
	main()
	