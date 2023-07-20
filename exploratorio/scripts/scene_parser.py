#!/usr/bin/env python3

__all__ = ["SubscribeRegress"]

import os
import sys
import time
import rospy
import laser_geometry.laser_geometry as lg
import pcl
import sensor_msgs
import logging
import pickle
import rospkg
import threading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sensor_msgs.msg import Image, LaserScan, PointCloud2
from message_filters import TimeSynchronizer, Subscriber, ApproximateTimeSynchronizer
# from geometry_msgs.msg import WrenchStamped

# Necessary to resolve deprecation issue with np aliases
# Must include before importing ros_numpy
np.float = float
import ros_numpy

from rospy.rostime import Duration
# from gazebo_msgs.srv import GetLinkState, GetLinkStateResponse
# from gazebo_msgs.srv import GetJointProperties, GetJointPropertiesResponse

from tf.transformations import euler_from_quaternion
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
# from scripts.wrench_viz import WrenchVisualizer

rospack = rospkg.RosPack()
# data_dir = rospack.get_path('param_est') + '/data'
# if not os.path.exists(data_dir):
# 	os.makedirs(data_dir)

LOGGER = logging.getLogger(__name__)

def image_to_numpy(img: Image) -> np.ndarray:
    return np.frombuffer(img.data, dtype=np.uint8)

def ros_pointcloud2_to_pcl(msg: PointCloud2) -> pcl.PointCloud:
    msg = ros_numpy.numpify(msg)
    points = np.zeros((msg.shape[0], 3))
    points[:, 0] = msg['x']
    points[:, 1] = msg['y']
    points[:, 2] = msg['z']
    return pcl.PointCloud(points.astype(np.float32))

class SceneParser():
    def __init__(self, rate, verbose=False, data_collect=False):
        super(SceneParser, self).__init__()

        self.first_log = True  # for debugging
        self.loaded_data = False

        self.verbose = verbose
        self.basetopic = "/prius"
        self.suffix_cam_topic = "_camera/image_raw"
        self.suffix_laser_topic = "_laser/scan"
        self.subNameImageLeft = self.basetopic + "/left" + self.suffix_cam_topic
        self.topics = []

        camera_keys = [f"{loc}_camera" for loc in ['front', 'back', 'left', 'right']]
        laser_keys = [f"{loc}_laser" for loc in ['front_left', 'front_right', 'center']]

        self.msg_data = {key: None for key in camera_keys + laser_keys}

        self.gravity_vec = np.array(([[0, 0, -9.8]])).T

        # if do_plot:
        #     fig = plt.figure(figsize=(16, 9))
        #     gs = gridspec.GridSpec(1, 1, figure=fig)
        #     plt.ion()

        #     # self.viz = WrenchVisualizer(fig, gs[0], pause_time=1e-4, labels=['Wrench: force, torque']
        #     # 								_fontdict=fontdict)

        # for data collection
        self.data_collect_mode = data_collect
        # if self.data_collect_mode:
        # 	joints = {"wrench_s": np.zeros((6,1)), "wrench_u":np.zeros((6,1)), "wrench_p": np.zeros((6,1)),
        # 		  "wrench_piston_butt": np.zeros((6,1))}
        # 	self.wrenches = {f"chain{i+1}":joints for i in range(6)}
        # 	self.data_collect_mode = data_collect
        # 	self.counter = 0

        self.laser_projector = lg.LaserProjection()

    def image_subscribers(self):
        return [
            Subscriber(rf"{self.basetopic}/{loc}{self.suffix_cam_topic}", Image)
            for loc in ['front', 'back', 'left', 'right']
        ]

    def scan_subs(self):
        laser_scans = [
            Subscriber(rf"{self.basetopic}/front_{loc}{self.suffix_laser_topic}", LaserScan)
            for loc in ['left', 'right']
        ]
        center_scan_sub = Subscriber(rf"{self.basetopic}/center{self.suffix_laser_topic}", LaserScan)
        return laser_scans + [center_scan_sub]

    def subscribers(self):
        cam_subs = self.image_subscribers()
        laser_subs = self.scan_subs()

        self.topics = [sub.name for sub in cam_subs + laser_subs]

        self.approx_ts = ApproximateTimeSynchronizer(cam_subs + laser_subs, 10, slop=0.51)
        self.approx_ts.registerCallback(self.cb)
        rospy.spin()

    def cb(self, *subscribers):
        """
        In order:
            front_camera
            back_camera
            left_camera
            right_camera
            front_left_laser
            front_right_laser
            center_laser
        """
        front_camera, back_camera, left_camera, right_camera = subscribers[:4]
        frontleft_laser, frontright_laser, center_laser = subscribers[4:]

        frontleft_cloud = self.laser_projector.projectLaser(frontleft_laser)
        frontright_cloud = self.laser_projector.projectLaser(frontright_laser)
        center_cloud = self.laser_projector.projectLaser(center_laser)

        self.msg_data['front_camera'] = front_camera
        self.msg_data['back_camera'] = back_camera
        self.msg_data['left_camera'] = left_camera
        self.msg_data['right_camera'] = right_camera

        self.msg_data['front_left_laser'] = frontleft_cloud
        self.msg_data['front_right_laser'] = frontright_cloud
        self.msg_data['center_laser'] = center_cloud
        self.loaded_data = True

        # frontleft_cloud_pcl = ros_pointcloud2_to_pcl(frontleft_cloud)
        # frontright_cloud_pcl = ros_pointcloud2_to_pcl(frontright_cloud)
        # center_cloud_pcl = ros_pointcloud2_to_pcl(center_cloud)

        # start = rospy.get_rostime().to_sec()
        # print(front_camera)
        # "process the intermediary twelve wrenches"
        # piston prismatic joints:: parent: cylinder, child: shaft
        # self.params["chain1"]["wrench_p"] = self.gather_wrench(p1_prism.wrench)
        # self.params["chain2"]["wrench_p"] = self.gather_wrench(p2_prism.wrench)


if __name__ == "__main__":
    rospy.init_node("estimator_node", anonymous=True, disable_signals=True)
    scene_parser = SceneParser(30, verbose=True)

    # collect some wrenches and 0plot
    try:
        while not rospy.is_shutdown():
            scene_parser.subscribers()
    except rospy.ROSInterruptException:
        rospy.logfatal("shutting down ros")
