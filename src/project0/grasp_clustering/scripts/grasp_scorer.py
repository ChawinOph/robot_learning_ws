#!/usr/bin/env python

import math
import numpy
import rospy
from sklearn.metrics.cluster import adjusted_rand_score

from grasp_clustering.msg import GraspInfo

class GraspScorer(object):

    def __init__(self):
        self.sub = rospy.Subscriber("/labeled_grasp_info", GraspInfo, self.callback)
        self.recv_glove_data = []
        self.recv_labels = []
        file = rospy.get_param('~test_filename')
        data = numpy.genfromtxt(fname=file, delimiter = ',', skip_header=1)
        self.true_labels = data[:,0]
        print "Loaded: " + str(self.true_labels.shape[0]) + " labels"
        
    def callback(self, msg):
        self.recv_glove_data.append(msg.glove)
        self.recv_labels.append(msg.label)
        if len(self.recv_labels)%10 == 0:
            print "Received: " + str(len(self.recv_labels)) + " grasps"
        if (len(self.recv_labels) == 1040):
            score = adjusted_rand_score(self.true_labels, self.recv_labels)
            print "Computed score: " + str(score)
            if (score > 0.72):
                print "PASS"
            else:
                print "FAIL"
            rospy.signal_shutdown("We are done.")
        
if __name__ == '__main__':
    rospy.init_node('grasp_scorer', anonymous=True)
    gs = GraspScorer()
    rospy.spin()
