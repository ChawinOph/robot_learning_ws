#!/usr/bin/env python

import numpy
import rospy
import time

from grasp_clustering.msg import GraspInfo

class GraspPublisher(object):

    def __init__(self):
        self.pub = rospy.Publisher("/grasp_info", GraspInfo, queue_size=1)
        time.sleep(1)

    def publish(self):
        file = rospy.get_param('~test_filename')
        data = numpy.genfromtxt(fname=file, delimiter = ',', skip_header=1)
        rate = rospy.Rate(100)
        for i in range(0,data.shape[0]):
            message = GraspInfo()
            message.label = -1
            message.emg = data[i,1:9]
            message.glove = data[i,9:24]
            self.pub.publish(message)
            rate.sleep()
            
        
if __name__ == '__main__':
    rospy.init_node('grasp_publisher', anonymous=True)
    gp = GraspPublisher()
    gp.publish()
    time.sleep(1)
    print "Publishing done."
        
