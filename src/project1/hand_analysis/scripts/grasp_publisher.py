#!/usr/bin/env python

import numpy
import rospy
import time

from hand_analysis.msg import GraspInfo

class GraspPublisher(object):

    def __init__(self):
        self.pub = rospy.Publisher("/grasp_info", GraspInfo, queue_size=100)
        time.sleep(1)

    def publish(self):
        file = rospy.get_param('~test_filename')
        data = numpy.genfromtxt(fname=file, delimiter = ',', skip_header=1)
        rate = rospy.Rate(200)

        #test glove to object
        for i in range(0,data.shape[0]):
            if rospy.is_shutdown(): return
            message = GraspInfo()
            message.id = -1
            message.label = -1
            message.glove = data[i,9:24]
            self.pub.publish(message)
            rate.sleep()
            
        #test emg to glove and object
        for i in range(0,data.shape[0]):
            if rospy.is_shutdown(): return
            message = GraspInfo()
            message.id = -1
            message.label = -1
            message.emg = data[i,1:9]
            self.pub.publish(message)
            rate.sleep()

        #test dimensionality reduction
        message = GraspInfo()
        message.glove_low_dim=[0,0]
        self.pub.publish(message)
        rate.sleep()
        message.glove_low_dim=[1,0]
        self.pub.publish(message)
        rate.sleep()
        message.glove_low_dim=[0,1]
        self.pub.publish(message)
        rate.sleep()
                    
if __name__ == '__main__':
    rospy.init_node('grasp_publisher', anonymous=True)
    gp = GraspPublisher()
    gp.publish()
    time.sleep(1)
    print "Publishing done."
        
