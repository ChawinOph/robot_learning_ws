#!/bin/bash

echo "Checking your solution ..."

roslaunch grasp_clustering project0.launch &

#roscore & 

#rosrun grasp_clustering cluster_grasps.py _train_filename:=data/object_grasping_30sec_no_labels.csv &

sleep 5

xterm -hold -e "rosrun grasp_clustering grasp_scorer.py _test_filename:=data/object_grasping_10sec.csv" & 

rosrun grasp_clustering grasp_publisher.py _test_filename:=data/object_grasping_10sec.csv

sleep 5

killall -9 rosmaster
killall python



