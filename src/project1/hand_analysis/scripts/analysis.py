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
from sklearn import svm # mainly with RBF: radial basis function
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process.kernels import RBF
from sklearn import linear_model
from sklearn.decomposition import PCA 

# Import messeges
from std_msgs.msg import String
from hand_analysis.msg import GraspInfo 


def custom_weights(dist_array):
	return np.array(dist_array)**-1

# create the node class
class Analysis():

	def __init__(self):
		# train_file_dir_tmp = rospack.get_path('hand_analysis') + '/' + rospy.get_param('~train_filename')
		# train_data = np.genfromtxt(fname=train_file_dir_tmp, delimiter=',', skip_header=1)
		train_data = np.genfromtxt(fname=rospy.get_param('~train_filename'), delimiter=',', skip_header=1)
		label_train_data = train_data[:, 0] 	# object labels
		emg_train_data = train_data[:, 1:9] 	# emg data
		glove_train_data = train_data[:, 9:]    # glove data
		print "Started trainings..."

		# train the object classification (clf := classiflier) being grasped based on glove data (classification)
		# 1st choic: SVM
		self.glove_to_label_clf = svm.SVC(gamma='scale').fit(glove_train_data, label_train_data)
		print "Finished glove to label classification training..."

		# train the object classification being grasped based on EMG data (classification)
		# 1st choice: svm.SVC(gamma='scale', kernel='rbf', C=1.0): (got score 0.4 < 0.62)
		# 2nd choice: svm.SVC(gamma='scale', kernel='rbf', C=10.0): (got score ~0.408 < 0.62)
		# 3rd choice: svm.SVC(gamma='auto', kernel='rbf', C=1.0): (got score ~0.143 < 0.62)
		# 4th choice: svm.SVC(gamma='auto', kernel='poly', C=1.0): (took too long to train and the grader exited)
		# 5th choice: KNeighborsClassifier(): (got score ~0.56 < 0.62)
		# 6th choice: KNeighborsClassifier(n_neighbors=10): (got score ~0.5744 < 0.62)
		# *7th choice: KNeighborsClassifier(n_neighbors=10, weights='distance', algorithm='ball_tree'/'kd_tree'): (got score ~0.5801 < 0.62)
		# 8th choice: KNeighborsClassifier(n_neighbors=10, weights='distance', p=3): (got score ~0.5744 < 0.62)
		# 9th choice: KNeighborsClassifier(n_neighbors=15, weights='distance'): (got score ~0.5744 < 0.62)
		# 10th choice: GaussianProcessClassifier(1.0 * RBF(1.0)): took too long
		# 11th choice: KNeighborsClassifier(n_neighbors=100) (got 0.407 < 0.62)
		# 12th choice: RadiusNeighborsClassifier(radius=500.0) (got 0.191)
		# 13th choice: NearestCentroid() (got 0.442)
		# 14th choice: KNeighborsClassifier(n_neighbors=10, weights=custom_weights) 1/dist^2 (score 0.579)
		# 15th choice: svm.SVC(gamma='scale', kernel='sigmoid', coef0=0) (score 0.1425)
		# **16th choice: LinearDiscriminantAnalysis() (score 0.5829)
		# ***17th choice: QuadraticDiscriminantAnalysis() (score 0.612)
		# ****18th choice: QuadraticDiscriminantAnalysis(reg_param=0.5) (score 0.6227)
		self.emg_to_label_clf = QuadraticDiscriminantAnalysis(reg_param=0.5)
		self.emg_to_label_clf.fit(emg_train_data, label_train_data)
		print "Finished EMG to label classification training..."

		# train to predict glove data based on EMG (regression)
		# 1st choice: Ordinary Linear Regression
		self.emg_to_glove_reg = linear_model.LinearRegression().fit(emg_train_data, glove_train_data)
		print "Finished EMG to glove regression training..."

		# Reduce the dimentionality of the glove data without sacrificing to much info
		# First, Generic PCA
		self.glove_dim_red = PCA(n_components=2).fit(glove_train_data)
		print "Finished glove dimensionality reduction..."

		print "Finished all trainings..."
		# set up subscirber for the incoming grasp_info 
		self.grasp_info_sub = rospy.Subscriber(name = "/grasp_info", data_class = GraspInfo, callback=self.callback, queue_size=100)
		# set up publisher for the /labeled_grasp_info
		self.labeled_grasp_info_pub = rospy.Publisher("/labeled_grasp_info", GraspInfo, queue_size=100)

	def callback(self, grasp_info):
		if np.array(grasp_info.glove).size != 0: # only glove data is given
			# print "Received testing glove data..."
			grasp_info.label = self.label_object_from_glove(grasp_info.glove)
			grasp_info.glove_low_dim = self.dim_red(grasp_info.glove)
		elif np.array(grasp_info.emg).size != 0: # only EMG is given
			# print "Received testing emg data..."
			grasp_info.label = self.label_object_from_emg(grasp_info.emg)
			grasp_info.glove = self.label_glove_from_emg(grasp_info.emg)
		elif np.array(grasp_info.glove_low_dim).size != 0: # only low-D glove data is given
			# print "Received testing glove_low_dim data..."
			grasp_info.glove = self.inv_dim_red(grasp_info.glove_low_dim)
			grasp_info.label = self.label_object_from_glove(grasp_info.glove)
		self.labeled_grasp_info_pub.publish(grasp_info)

	def label_object_from_glove(self, recv_glove):
		return self.glove_to_label_clf.predict([recv_glove])[0]

	def label_object_from_emg(self, recv_emg):
		return self.emg_to_label_clf.predict([recv_emg])[0]

	def label_glove_from_emg(self, recv_emg):
		return self.emg_to_glove_reg.predict([recv_emg])[0]

	def dim_red(self, recv_glove):
		return self.glove_dim_red.transform([recv_glove])[0]

	def inv_dim_red(self, recv_glove_low_dim):
		return self.glove_dim_red.inverse_transform([recv_glove_low_dim])[0]

if __name__ == '__main__':
	rospy.init_node('analysis', anonymous=True)
	print "Initialized 'analysis' node" 
	try:
		obj = Analysis()
	except rospy.ROSInterruptException:
		pass
	rospy.spin()
	print "Spinning..."

	