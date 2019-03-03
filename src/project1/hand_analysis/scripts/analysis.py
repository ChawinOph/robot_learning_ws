#!/usr/bin/env python

##########################################
##### WRITE YOUR CODE IN THIS FILE #######
##########################################
        
import rospy  # for using python in ROS environment
import rospkg # for finding directory of current package
rospack = rospkg.RosPack() # get an instance of RosPack with the default search paths
import numpy as np 
np.set_printoptions(threshold=np.inf) # change the print-out option for large array size

# import scikit learn modules
from sklearn.cluster import KMeans # for using K-mean clustering algorithm
from sklearn.decomposition import PCA 

# Import messeges
from std_msgs.msg import String
from hand_analysis.msg import GraspInfo 

# create the node class
class Analysis():

	def __init__(self):
		train_file_dir_tmp = rospack.get_path('hand_analysis') + '/' + rospy.get_param('~train_filename')
		train_data = np.genfromtxt(fname=train_file_dir_tmp, delimiter=',', skip_header=1)
		clas_gt_train_data = train_data[:, 0] 	# object labels
		emg_train_data = train_data[:, 1:9] 	# emg data
		glove_train_data = train_data[:, 9:]    # glove data
		print "Started Training..."

		# train the object classification being grasped based on glove data (classification)
		# 1st choic: SVM

		# train the object classification being grasped based on EMG data (classification)
		# 1st choice: SVM 

		# train to predict glove data baesd on EMG (regression)
		# 1st choice: Ordinary Linear Regression

		# Reduce the dimentionality of the glove data without sacrificing to much info
		# First, try the generic PCA
		self.pca = PCA(n_components=2)
		self.pca.fit(glove_train_data)
		print "Finished Training..."
		# set up subscirber for the incoming grasp_info 
		self.grasp_info_sub = rospy.Subscriber(name = "/grasp_info", data_class = GraspInfo, callback=self.callback, queue_size=100)
		# set up publisher for the /labeled_grasp_info
		self.labeled_grasp_info_pub = rospy.Publisher("/labeled_grasp_info", GraspInfo, queue_size=100)

	def callback(self, grasp_info):
		print "Received testing grasp data..."
		if np.array(grasp_info.glove).size != 0: # only glove data is given
			grasp_info.label = self.label_object_from_glove(grasp_info.glove)
			grasp_info.glove_low_dim = self.dim_red(grasp_info.glove)
		elif np.array(grasp_info.emg).size != 0: # only EMG is given
			grasp_info.label = self.label_from_emg(grasp_info.emg)
			grasp_info.glove = self.label_glove_from_emg(grasp_info.emg)
		elif np.array(grasp_info.glove_low_dim).size != 0: # only low-D glove data is given
			grasp_info.glove = self.inv_dim_red(grasp_info.glove_low_dim)
			grasp_info.label = self.label_object_from_glove(grasp_info.glove_low_dim)

	def label_object_from_glove(self, recv_glove):
		return # non numpy data type

	def label_object_from_emg(self, recv_emg):
		return

	def label_glove_from_emg(self, recv_emg):
		return

	def dim_red(self, recv_glove):
		return

	def inv_dim_red(self, recv_glove_low_dim):
		return

if __name__ == '__main__':
	rospy.init_node('analysis', anonymous=True) 
	try:
		obj = Analysis()
	except rospy.ROSInterruptException:
		pass
	rospy.spin()
	print "Spinning..."

	