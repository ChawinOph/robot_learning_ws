#!/usr/bin/env python

import math
import numpy
import rospy
from sklearn.metrics.cluster import adjusted_rand_score

from hand_analysis.msg import GraspInfo

def compare(x,y):
    assert(len(x) == len(y))
    correct = float(0)
    for i in range(0,len(x)):
        if x[i] == y[i]: correct = correct+1
    return correct/len(x)

def regression_compare(x,y):
    assert(len(x) == len(y))
    error = 0.0
    for i in range(0,len(x)):
        error = error + numpy.linalg.norm( x[i,:] - y[i,:] )
    return error / len(x)

class GraspScorer(object):

    def __init__(self):
        file = rospy.get_param('~test_filename')
        data = numpy.genfromtxt(fname=file, delimiter = ',', skip_header=1)
        self.true_labels = data[:,0]
        self.true_glove = data[:,9:24]
        print "Loaded: " + str(self.true_labels.shape[0]) + " test data points."
        self.state='GLOVE_TO_OBJ'
        self.transition = True
        self.score = 0
        self.sub = rospy.Subscriber(name = "/labeled_grasp_info", data_class = GraspInfo,
                                    callback = self.callback, queue_size = 100)

    def score_glove_to_object(self, msg):
        if (self.transition):
            self.recv_labels = []
            self.transition = False
        self.recv_labels.append(msg.label)
        if len(self.recv_labels)%100 == 0:
            print "Scoring glove to object. Received: " \
                + str(len(self.recv_labels)) + " messages."
        if len(self.recv_labels) == self.true_labels.shape[0]:
            score = compare(self.true_labels, self.recv_labels)
            print "Glove to object score: "  + str(score)
            if score > 0.97: points = 3
            else: points = 0
            print "Points received: " + str(points) + "/3"
            self.score = self.score + points
            self.transition = True
            self.state='EMG_TO_GLOVE_OBJ'

    def score_emg_to_glove_object(self, msg):
        if (self.transition):
            self.recv_labels = []
            self.recv_glove = numpy.empty_like(self.true_glove)
            self.transition = False
        self.recv_glove[ len(self.recv_labels), : ] = msg.glove
        self.recv_labels.append(msg.label)
        if len(self.recv_labels)%100 == 0:
            print "Scoring EMG to glove and object. Received: " \
                + str(len(self.recv_labels)) + " messages."
        if len(self.recv_labels) == self.true_labels.shape[0]:
            score = compare(self.true_labels, self.recv_labels)
            print "EMG to object classification score: "  + str(score)
            if score > 0.62: points = 4
            else: points = 0
            print "Points received: " + str(points) + "/4"
            self.score = self.score + points

            score = regression_compare(self.true_glove, self.recv_glove)
            print "EMG to glove regression score: "  + str(score)            
            if score < 0.27: points = 4
            else: points = 0
            print "Points received: " + str(points) + "/4"
            self.score = self.score + points
            self.transition = True
            self.state='GLOVE_DIM_RED'

    def score_glove_dim_red(self,msg):
        if (self.transition):
            self.count = 0
            self.transition = False
        if self.count == 0:
            self.origin = numpy.asarray(msg.glove)
            self.count = self.count+1
            print "Scoring glove dimensionality reduction. Origin received."
            return
        elif self.count == 1:
            self.e1 = numpy.asarray(msg.glove)
            self.count = self.count+1
            print "Scoring glove dimensionality reduction. First basis vector received."
            return
        self.e2 = numpy.asarray(msg.glove)
        print "Scoring glove dimensionality reduction. Second basis vector received."
        self.e1 = self.e1 - self.origin
        self.e2 = self.e2 - self.origin
        pred_glove = numpy.empty_like(self.true_glove)
        for i in range(0,self.true_glove.shape[0]):
            glove = self.true_glove[i,:]
            glove_rec = numpy.dot(glove-self.origin, self.e1) * self.e1 + \
                        numpy.dot(glove-self.origin, self.e2) * self.e2 + \
                        self.origin
            pred_glove[i,:] = glove_rec
        score = regression_compare(self.true_glove, pred_glove)
        print "Dimensionality reduction score: "  + str(score)                               
        if score < 0.18: points = 4
        else: points = 0
        print "Points received: " + str(points) + "/4"
        self.score = self.score + points
        self.transition = True
        self.state='DONE'
        self.finalize_score()

    def finalize_score(self):
        print "Final score: " + str(self.score) + "/15 points"
        rospy.signal_shutdown("We are done.")
            
    def callback(self, msg):
        if self.state=='GLOVE_TO_OBJ':
            self.score_glove_to_object(msg)
        elif self.state=='EMG_TO_GLOVE_OBJ':
            self.score_emg_to_glove_object(msg)
        elif self.state=='GLOVE_DIM_RED':
            self.score_glove_dim_red(msg)
        elif self.state=='DONE':
            print "Scoring finished"
            rospy.signal_shutdown("We are done.")
        else: print "Unknown state"
            
if __name__ == '__main__':
    rospy.init_node('grasp_scorer', anonymous=True)
    gs = GraspScorer()
    rospy.spin()
