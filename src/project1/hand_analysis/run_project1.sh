#!/bin/bash

echo "Checking your solution ..."

roslaunch hand_analysis project1.launch &

sleep 5

xterm -hold -e "rosrun hand_analysis grasp_scorer.py _test_filename:=data/object_grasping_10sec_no_labels_with_gt.csv" & 

rosrun hand_analysis grasp_publisher.py _test_filename:=data/object_grasping_10sec_no_labels_with_gt.csv

sleep 5

killall -9 rosmaster
killall python



