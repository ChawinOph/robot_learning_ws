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
		# create service caller from real_robot
		self.real_robot_action = rospy.ServiceProxy('real_robot', RobotAction)
		# publisher for gui
		self.pub = rospy.Publisher("/robot_states", RobotState, queue_size=100)
		self.num_tests = 50		# default: 21
		self.perturb_steps = 200    # default: 200
		print "Collecting data from real_robot..."
		self.features = [];
		self.labels = [];
		start_time = time.time()
		self.obtain_data()
		self.elapsed_collecting_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
		print "Finished collecting data"
		print "Training the network..."
		start_time = time.time()
		self.network = MyDNN(self.features.shape[1], self.labels.shape[1])
		self.trainer = MyDNNTrain(self.network)
		self.trainer.train(self.labels, self.features)
		self.elapsed_training_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
		print "Finished training fake_robot"
		# create fake robot service
		self.fake_robot_service = rospy.Service('fake_robot', RobotAction, self.fake_robot_action)
		self.resp_fake = RobotActionResponse()
		# reset the robot_state in response
		self.resp_fake.robot_state = self.features[0, :6].tolist

	def obtain_data(self):
		for i in range(0, self.num_tests):
			action = np.random.rand(1,3)
			action[0,0] = (2 * action[0,0] - 1.0) * 1.0
			action[0,1] = (2 * action[0,1] - 1.0) * 0.5
			action[0,2] = (2 * action[0,2] - 1.0) * 0.25
			self.perturb(action.reshape(3))
		# convert lists to numpy arrays
		self.features = np.array(self.features)
		self.labels = np.array(self.labels)
		
	def perturb(self, action):
		req_real = RobotActionRequest()
		req_real.reset = True
		# reset robot config
		resp_real = self.real_robot_action(req_real)
		# apply constant action
		for j in range(self.perturb_steps):
			req_real = RobotActionRequest()
			req_real.reset = False
			req_real.action = action
			self.features.append(np.append(resp_real.robot_state, req_real.action).tolist())
			resp_real = self.real_robot_action(req_real)
			self.labels.append(resp_real.robot_state)
			# visualizing the perturbed real_robot in gui
			self.viz_robot('real_robot', resp_real.robot_state)
			# time.sleep(0.04) 

	def viz_robot(self, robot_name, robot_state):
		message = RobotState()
		message.robot_name=str(robot_name)
		message.robot_state = robot_state
		self.pub.publish(message)

	def fake_robot_action(self, req_fake):
		# if the request has the reset as true, reset the fake_robot config
		if req_fake.reset:			
			self.resp_fake.robot_state = self.features[0, :6].tolist()	
		else:
			self.resp_fake.robot_state = self.network.predict(np.append(self.resp_fake.robot_state, req_fake.action)).tolist()
		return self.resp_fake

class MyDNN(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(MyDNN, self).__init__()
		hl1_n_nodes = 64 
		self.fc1 = nn.Linear(input_dim, hl1_n_nodes)
		self.fc2 = nn.Linear(hl1_n_nodes, hl1_n_nodes) # hidden layer 1
		self.fc3 = nn.Linear(hl1_n_nodes, output_dim)

		# 2 hidden layers
		# hl1_n_nodes = 32 
		# hl2_n_nodes = 32 
		# self.fc1 = nn.Linear(input_dim, hl1_n_nodes)
		# self.fc2 = nn.Linear(hl1_n_nodes, hl2_n_nodes) # hidden layer 1
		# self.fc3 = nn.Linear(hl2_n_nodes, hl2_n_nodes) # hidden layer 2
		# self.fc4 = nn.Linear(hl2_n_nodes, output_dim)

	def forward(self, x):
		# 1 hidden layer
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)

		# 2 hidden layers
		# x = F.relu(self.fc1(x))
		# x = F.relu(self.fc2(x))
		# x = F.relu(self.fc3(x))
		# x = self.fc4(x)

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
		self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.learning_rate) # default: torch.optim.SGD(self.network.parameters(), lr=self.learning_rate)
		self.criterion = nn.MSELoss() # default: nn.MSELoss()
		self.num_epochs = 200	# default: 500
		self.batchsize = 25	# default: 100
		self.shuffle = False 	# default: True

	def train(self, labels, features):
		self.network.train()
		dataset = MyDataset(labels, features)
		loader = DataLoader(dataset, shuffle=self.shuffle, batch_size = self.batchsize)
		for epoch in range(self.num_epochs):
			print 'epoch ', (epoch + 1)
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
		print 'loss ', total_loss/i

def main():
	rospy.init_node('fake_robot', anonymous=True)
	start_time = time.time()
	fake_robot = FakeRobot();
	elapsed_time = time.time() - start_time
	print "collection time: " + fake_robot.elapsed_collecting_time
	print "training time: " + fake_robot.elapsed_training_time
	print "total time: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
	print "Fake robot now spinning"
	rospy.spin()

if __name__ == '__main__':
	main()
	