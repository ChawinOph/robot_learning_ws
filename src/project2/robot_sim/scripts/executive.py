#!/usr/bin/env python

import numpy as np
import time
import rospy

from robot_sim.msg import RobotState
from robot_sim.srv import RobotAction
from robot_sim.srv import RobotActionRequest
from robot_sim.srv import RobotActionResponse

def wrap(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi

class RobotExecutive(object):
    def __init__(self, num_random_tests, num_steps):
        self.num_random_tests = num_random_tests
        self.num_steps = num_steps
        self.pub = rospy.Publisher("/robot_states", RobotState, queue_size=100)
        self.real_robot_action = rospy.ServiceProxy('real_robot', RobotAction)
        self.fake_robot_action = rospy.ServiceProxy('fake_robot', RobotAction)
    
    def score_state(self, true_state, predicted_state):
        true_pos = true_state[0:3]
        pred_pos = predicted_state[0:3]
        error = wrap(np.subtract(true_pos, pred_pos))
        # print error
        return np.linalg.norm(error) #Euclidean distance between position vectors
        #return np.linalg.norm(np.subtract(true_pos, pred_pos))

    def print_final_score(self):
        score = np.median(self.scoring_matrix)
        print "\nOverall median score: ",score
        if score<0.6:
            print "Final score: 20/20 points"
        elif score<0.8:
            print "Final score: 17/20 points"
        elif score<1.0:
            print "Final score: 14/20 points"
        elif score<1.5:
            print "Final score: 10/20 points"
        else:
            print "Final score: 5/20 points"

    def run(self):
        self.scoring_matrix = []
        for k in range(0,self.num_random_tests):
            action = np.random.rand(1,3)
            action[0,0] = (2 * action[0,0] - 1.0) * 1.0
            action[0,1] = (2 * action[0,1] - 1.0) * 0.5
            action[0,2] = (2 * action[0,2] - 1.0) * 0.25
            score = self.single_test(action)
            print "Score: ",score
            self.scoring_matrix.append(score)
            #time.sleep(1.00)

    def single_test(self, action):
        req = RobotActionRequest()
        req.reset = True
        resp = self.real_robot_action(req)
        resp = self.fake_robot_action(req)
        for i in range(self.num_steps):
            req = RobotActionRequest()
            req.reset = False
            req.action = action.reshape((3))
            resp_real = self.real_robot_action(req)
            message = RobotState()
            message.robot_name=str('real_robot')
            message.robot_state = resp_real.robot_state
            self.pub.publish(message)
            resp_fake = self.fake_robot_action(req)
            message = RobotState()
            message.robot_name=str('fake_robot')
            message.robot_state = resp_fake.robot_state
            self.pub.publish(message)
            #time.sleep(0.01)
        return self.score_state(resp_real.robot_state, resp_fake.robot_state)
 
if __name__ == '__main__':
    rospy.init_node('robot_executive', anonymous=True)
    executive = RobotExecutive(21,200)
    executive.run()
    time.sleep(1)
    executive.print_final_score()
    print "Executive done!"
