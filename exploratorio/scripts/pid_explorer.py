#!/usr/bin/env python3

import rospy
import laser_geometry.laser_geometry as lg
import pcl

from gazebo_msgs.msg import ModelStates
from message_filters import Subscriber, ApproximateTimeSynchronizer

import numpy as np
# Necessary to resolve deprecation issue with np aliases
# Must include before importing ros_numpy
np.float = float
import ros_numpy

from scene_parser import SceneParser

class PIDExplorer(SceneParser):
    def __init__(self, rate, **kwargs):
        super().__init__(rate, **kwargs)

    def subscribers(self):
        laser_subs = self.scan_subs()
        pos_sub = Subscriber("/gazebo/model_states", ModelStates)

        Subscriber("/gazebo/set_model_state", ModelState)

        self.approx_ts = ApproximateTimeSynchronizer(laser_subs + [pos_sub], 10, slop=0.51)
        self.approx_ts.registerCallback(self.cb)

        rospy.spin()
