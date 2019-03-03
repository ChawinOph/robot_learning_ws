#!/usr/bin/env python

##########################################
##### WRITE YOUR CODE IN THIS FILE #######
##########################################

import rospy  # for using python in ROS environment
import rospkg # for finding directory of current package
rospack = rospkg.RosPack() # get an instance of RosPack with the default search paths
import numpy as np 
np.set_printoptions(threshold=np.inf) # change the print-out option for large array size
from sklearn.cluster import KMeans # for using K-mean clustering algorithm
# (link: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

# Import messeges
from std_msgs.msg import String
from grasp_clustering.msg import GraspInfo 

# create the class node 
class Cluster_grasps():
	
	def __init__(self):
		# Get the ~private namespace parameters from command line or launch file.
		# train_file_dir = rospack.get_path('grasp_clustering') + '/' + rospy.get_param('~train_filename')
		train_file_dir = rospy.get_param('~train_filename')

		# get grasp_training_data from _train_filename 
		csv_train_data = np.genfromtxt(fname=train_file_dir, delimiter = ',', skip_header=1)
		# print csv_train_data
		# print csv_train_data.dtype # float64
		# print csv_train_data.shape # (2820, 23)

		# get the glove joint angle training data
		glove_train_data = csv_train_data[:, 8::]

		# fitting a K-means clustering algorithm to the glove (joint angle)
		self.kmeans = KMeans(n_clusters=10).fit(glove_train_data)
		print 'Finished clustering...'
		# print kmeans.labels_ # see the results

		# subscribe to the ROS topic called "/grasp_info"
		self.grasp_info_sub = rospy.Subscriber("/grasp_info", GraspInfo, self.callback)
		self.labeled_grasp_info_pub = rospy.Publisher("/labeled_grasp_info", GraspInfo, queue_size=1)
							
	def callback(self, grasp_info):
		# Assign a cluster label based on the .glove data, using the trained k-means
		grasp_info.label = self.kmeans.predict(np.array(grasp_info.glove).reshape(1, -1))
		# Re-publish it on the "/labeled_grasp_info" topic
		self.labeled_grasp_info_pub.publish(grasp_info)

if __name__ == '__main__':
	# Initialize the node: is very important as it tells rospy the name of your node, 
	# otherwise it cannot start communicating with the ROS Master.
	rospy.init_node('grasp_clustering', anonymous=True)
	
	try:
		obj = Cluster_grasps()
	except rospy.ROSInterruptException:
		pass
	
	# spin() simply keeps python from exiting until this node is stopped
	rospy.spin()
	print "Labelling done."
