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

import matplotlib.pyplot as plt

class FakeRobot(object):
	def __init__(self):
		# wait unitil the real_robot node is available
		rospy.wait_for_service('real_robot')
		# create service caller from real_robot
		self.real_robot_action = rospy.ServiceProxy('real_robot', RobotAction)
		# publisher for gui
		self.pub = rospy.Publisher("/robot_states", RobotState, queue_size=100)
		self.num_tests = 1500		# default: 21
		self.num_validate_tests = 100 
		self.perturb_steps = 200    # default: 200
		self.features = [];
		self.labels = [];
		self.valid_features = [];
		self.valid_labels = [];
		start_time = time.time()
		self.obtain_data()
		self.elapsed_collecting_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
		print "Finished collecting data"
		print "features shape: " 
		print self.features.shape
		print self.features
		print "labels shape: " 
		print self.labels.shape
		print self.labels
		print "valid_features shape: " 
		print self.valid_features.shape
		# print self.valid_features
		print "valid labels shape: " 
		print self.valid_labels.shape
		# print self.valid_labels
		print "Training the network..."
		start_time = time.time()
		self.network = MyDNN(self.features.shape[1], self.labels.shape[1])
		print self.network
		# create fake robot service
		self.fake_robot_service = rospy.Service('fake_robot', RobotAction, self.fake_robot_action)
		self.resp_fake = RobotActionResponse()
		# reset the robot_state in response
		self.resp_fake.robot_state = self.features[0, :6].tolist
		# train the network
		self.trainer = MyDNNTrain(self.network)
		self.trainer.train(self.labels, self.features, self.valid_labels, self.valid_features)
		self.elapsed_training_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
		print "Finished training fake_robot"
		n_epoch = np.arange(self.trainer.num_epochs) + 1
		plt.figure(1)
		plt.subplot(211)
		plt.plot(n_epoch, self.trainer.training_loss, n_epoch, self.trainer.valid_loss)
		# plt.xlabel('no. epochs')
		plt.legend(('training loss', 'validation loss'), loc='upper right')
		plt.title('Validation Curve')
		plt.subplot(212)
		plt.plot(n_epoch, self.trainer.generalization_loss, 'g', label='Generalization Loss (%)')
		plt.xlabel('no. epochs')
		# plt.title('Generalization Loss (%)')
		plt.legend(loc='upper right')
		plt.show()

	def obtain_data(self):
		# # create 3d mesh grid of all torque
		# tau_range = np.linspace(-1.0, 1.0, num=self.num_tests)
		# tau1v, tau2v, tau3v = np.meshgrid(tau_range, 0.5*tau_range, 0.25*tau_range, sparse=False, indexing='ij')
		# for i in range(self.num_tests):
		# 	for j in range(self.num_tests):
		# 		for k in range(self.num_tests):
		# 			print "i=%d j=%d k=%d" %(i,j,k)
		# 			action = np.array([tau1v[i, j, k], tau2v[i, j, k], tau3v[i, j, k]])
		# 			self.perturb(action)

		print "Collecting training data points from real_robot..."
		for i in range(0, self.num_tests):
			action = np.random.rand(1,3)
			action[0,0] = (2 * action[0,0] - 1.0) * 1.0
			action[0,1] = (2 * action[0,1] - 1.0) * 0.5
			action[0,2] = (2 * action[0,2] - 1.0) * 0.25
			self.perturb(action.reshape(3))
		# convert lists to numpy arrays
		self.features = np.array(self.features)
		self.labels = np.array(self.labels)

		# obtain validation data point
		print "Collecting validation data points from real_robot..." 
		for i in range(0, self.num_validate_tests):
			action = np.random.rand(1,3)
			action[0,0] = (2 * action[0,0] - 1.0) * 1.0
			action[0,1] = (2 * action[0,1] - 1.0) * 0.5
			action[0,2] = (2 * action[0,2] - 1.0) * 0.25
			self.valid_perturb(action.reshape(3))
		# convert lists to numpy arrays
		self.valid_features = np.array(self.valid_features)
		self.valid_labels = np.array(self.valid_labels)

		
	def perturb(self, action):
		req_real = RobotActionRequest()
		req_real.reset = True
		# send request to reset real_robot config
		resp_real = self.real_robot_action(req_real)
		# apply a constant action
		for j in range(self.perturb_steps):
			# create a new request
			req_real = RobotActionRequest()
			req_real.reset = False
			req_real.action = action
			self.features.append(np.append(resp_real.robot_state, req_real.action).tolist())
			# send request to move real_robot
			resp_real = self.real_robot_action(req_real)
			self.labels.append(resp_real.robot_state)
			# visualizing the perturbed real_robot
			# self.viz_robot('real_robot', resp_real.robot_state)

	def valid_perturb(self, action):
		req_real = RobotActionRequest()
		req_real.reset = True
		# send request to reset real_robot config
		resp_real = self.real_robot_action(req_real)
		for j in range(self.perturb_steps):
			# create a new request
			req_real = RobotActionRequest()
			req_real.reset = False
			req_real.action = action
			# collect only the last data point 
			if j == self.perturb_steps - 1:	
				self.valid_features.append(np.append(resp_real.robot_state, req_real.action).tolist())
			# send request to move real_robot
			resp_real = self.real_robot_action(req_real)
			if j == self.perturb_steps - 1:	
				self.valid_labels.append(resp_real.robot_state)

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

		hl1_n_nodes = 12
		self.fc1 = nn.Linear(input_dim, hl1_n_nodes)
		# self.drop1 = nn.Dropout(p=0.5)
		self.fc2 = nn.Linear(hl1_n_nodes, hl1_n_nodes) # hidden layer 1
		self.drop2 = nn.Dropout(p=0.25)
		self.fc3 = nn.Linear(hl1_n_nodes, output_dim)

		# 2 hidden layers
		# hl1_n_nodes = 15 
		# hl2_n_nodes = 15
		# drop_out_rate = 0.2
		# self.fc1 = nn.Linear(input_dim, hl1_n_nodes)
		# # self.drop1 = nn.Dropout(p=drop_out_rate)
		# self.fc2 = nn.Linear(hl1_n_nodes, hl2_n_nodes) # hidden layer 1
		# # self.drop2 = nn.Dropout(p=drop_out_rate)
		# self.fc3 = nn.Linear(hl2_n_nodes, hl2_n_nodes) # hidden layer 2
		# # self.drop3 = nn.Dropout(p=drop_out_rate)
		# self.fc4 = nn.Linear(hl2_n_nodes, output_dim)

		# 3 hidden layers
		# hl1_n_nodes = 16 
		# hl2_n_nodes = 8 
		# hl3_n_nodes = 16 
		# self.fc1 = nn.Linear(input_dim, hl1_n_nodes)
		# self.drop1 = nn.Dropout(p=0.1)
		# self.fc2 = nn.Linear(hl1_n_nodes, hl2_n_nodes) # hidden layer 1
		# self.drop2 = nn.Dropout(p=0.1)
		# self.fc3 = nn.Linear(hl2_n_nodes, hl3_n_nodes) # hidden layer 2
		# self.drop3 = nn.Dropout(p=0.1)
		# self.fc4 = nn.Linear(hl3_n_nodes, hl3_n_nodes) # hidden layer 3
		# self.drop4 = nn.Dropout(p=0.1)
		# self.fc5 = nn.Linear(hl3_n_nodes, output_dim)

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

		# 3 hidden layers
		# x = F.relu(self.fc1(x))
		# x = F.relu(self.fc2(x))
		# x = F.relu(self.fc3(x))
		# x = F.relu(self.fc4(x))
		# x = self.fc5(x)

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
		self.learning_rate = 0.005 # default: 0.01
		self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.learning_rate) # default: torch.optim.SGD(self.network.parameters(), lr=self.learning_rate)
		# self.optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
		self.criterion = nn.MSELoss() # default: nn.MSELoss()
		self.num_epochs = 200	# default: 500
		self.batchsize = 30		# default: 100
		self.shuffle = True # default: True
		self.current_loss_change = 1 # for tracking the loss changes between epochs
		self.current_loss = 1 # for tracking the current loss
		self.training_loss = np.zeros(self.num_epochs)
		self.current_valid_loss = 1		 # for tracking the current loss
		self.current_valid_loss_change = 1	# for tracking the current loss
		self.valid_loss = np.zeros(self.num_epochs)
		self.loss_threshold = 0.003
		self.loss_change_threshold = 0.0001
		"""generalization loss: https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf
		the relative increase of the validation error over the minimum so far (in percent)"""
		self.generalization_loss = np.zeros(self.num_epochs)

	def train(self, labels, features, valid_labels, valid_features):
		self.network.train()
		dataset = MyDataset(labels, features)
		valid_dataset = MyDataset(valid_labels, valid_features)
		loader = DataLoader(dataset, shuffle=self.shuffle, batch_size = self.batchsize)
		valid_loader = DataLoader(valid_dataset, shuffle=self.shuffle, batch_size = self.batchsize)
		for epoch in range(self.num_epochs):
			print 'epoch ', (epoch + 1)
			# if self.current_loss_change > 0.0001:
			# 	self.learning_rate = 0.005 
			# else:
			# 	self.learning_rate = 0.005/2.0
			# 	self.loss_change_threshold = 0.00001
			# elif self.current_loss_change > 0.00005:
			# 	self.learning_rate = 0.005/4.0 
			# else:
			# 	self.learning_rate = 0.005/8.0 
			# print "current learning rate: %f" %self.learning_rate
			# self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.learning_rate) 
			self.train_epoch(loader)
			self.valid_epoch(valid_loader)
			self.training_loss[epoch] = self.current_loss
			self.valid_loss[epoch] = self.current_valid_loss
			self.generalization_loss[epoch] = 100*((self.valid_loss[epoch]/np.amin(self.valid_loss[:(epoch+1)])) - 1.0)
			print 'generalization_loss', self.generalization_loss[epoch]
			# if self.current_loss < self.loss_threshold or self.current_loss_change < self.loss_change_threshold:
			# 	print "Reached the loss threshold"
			# 	break

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
		self.current_loss_change = self.current_loss - total_loss/i
		self.current_loss = total_loss/i
		print 'loss ', total_loss/i
		print 'loss_change ', self.current_loss_change 

	def valid_epoch(self, valid_loader):
		total_loss = 0.0
		for i, data in enumerate(valid_loader):
			features = data['feature'].float()
			labels = data['label'].float()
			# self.optimizer.zero_grad()
			predictions = self.network(features)
			loss = self.criterion(predictions, labels)
			# loss.backward()
			total_loss += loss.item()
			# self.optimizer.step()	
		self.current_valid_loss_change = self.current_valid_loss - total_loss/i
		self.current_valid_loss = total_loss/i
		print 'valid loss ', total_loss/i
		print 'valid_loss_change ', self.current_valid_loss_change 

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
	