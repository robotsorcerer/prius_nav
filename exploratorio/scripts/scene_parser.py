#!/usr/bin/env python3

import os
import sys
import rospy
import laser_geometry.laser_geometry as lg
import pcl
import logging
import rospkg
import numpy as np
from sensor_msgs.msg import Image, LaserScan, PointCloud2
from message_filters import Subscriber, ApproximateTimeSynchronizer
from typing import List
# from geometry_msgs.msg import WrenchStamped

# Necessary to resolve deprecation issue with np aliases
# Must include before importing ros_numpy
np.float = float
import ros_numpy

from enum import IntFlag, auto
from rospy.rostime import Duration

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from scripts.wrench_viz import WrenchVisualizer

rospack = rospkg.RosPack()
# data_dir = rospack.get_path('param_est') + '/data'
# if not os.path.exists(data_dir):
# 	os.makedirs(data_dir)

LOGGER = logging.getLogger(__name__)


def ros_pointcloud2_to_pcl(msg: PointCloud2) -> pcl.PointCloud:
    """
    Converts a `sensor_msgs.msg.PointCloud2` message into a `PointCloud` pcl object.
    """
    msg = ros_numpy.numpify(msg)
    points = np.zeros((msg.shape[0], 3))
    points[:, 0] = msg['x']
    points[:, 1] = msg['y']
    points[:, 2] = msg['z']
    return pcl.PointCloud(points.astype(np.float32))

class SensorSource(IntFlag):
    """
    This enum represents the different sensors that we will collect data from.
    These values will be used as keys to disambiguate the sources of data in
    the `SceneParser`.
    """
    FRONT_CAMERA = auto()
    BACK_CAMERA = auto()
    LEFT_CAMERA = auto()
    RIGHT_CAMERA = auto()
    ALL_CAMERAS = FRONT_CAMERA | BACK_CAMERA | LEFT_CAMERA | RIGHT_CAMERA

    FRONT_LEFT_LASER = auto()
    FRONT_RIGHT_LASER = auto()
    CENTER_LASER = auto()
    ALL_LASERS = FRONT_LEFT_LASER | FRONT_RIGHT_LASER | CENTER_LASER

    ALL_SENSORS = ALL_CAMERAS | ALL_LASERS

class SceneParser():
    """
    This ROS node collects data from various sensors in the prius robot.

    It subscribes to /prius/{front,back,left,right}_camera/image_raw to
    collect image data, and subscribes to /prius/{front_left,front_right,center}_laser/scan
    for laser scan data. Upon receiving data from these topics, the data is
    stored in `msg_data[sensor_enum]` where `sensor_enum` is `SensorSource` value
    corresponding to the relevant sensor.
    """

    def __init__(self, rate, verbose=False, data_collect=False):
        """
        rate: integer, not currently used
        """
        super(SceneParser, self).__init__()
        self.first_log = True  # for debugging
        self.loaded_data = False

        self.verbose = verbose
        self.basetopic = "/prius"
        self.suffix_cam_topic = "_camera/image_raw"
        self.suffix_laser_topic = "_laser/scan"
        self.subNameImageLeft = self.basetopic + "/left" + self.suffix_cam_topic
        self.topics = []

        self.msg_data = {key: None for key in SensorSource if not key.name.startswith("ALL")}

        """
        This is required in order to translate a ROS LaserScan msg into a
        pcl PointCloud2 object.
        """
        self.laser_projector = lg.LaserProjection()

    def image_subscribers(self) -> List[str]:
        """
        Returns the list of raw image topics to subscribe to.
        We subscribe to the raw images from all camera topics.
        """
        return [
            Subscriber(rf"{self.basetopic}/{loc}{self.suffix_cam_topic}", Image)
            for loc in ['front', 'back', 'left', 'right']
        ]

    def scan_subs(self) -> List[str]:
        """
        Returns the list of laser topics to subscribe to.
        We subscribe to the laser scans from all lasers.
        """
        laser_scans = [
            Subscriber(rf"{self.basetopic}/front_{loc}{self.suffix_laser_topic}", LaserScan)
            for loc in ['left', 'right']
        ]
        center_scan_sub = Subscriber(f"{self.basetopic}/center{self.suffix_laser_topic}", LaserScan)
        return laser_scans + [center_scan_sub]

    def subscribers(self):
        """
        Instantiates the Subscribers for the topics of interest, registers
        the callback, and spins ROS.

        Since we are collecting data from a multitude of sensors, we collect all
        subscribers in a ApproximateTimeSynchronizer message filter, so we will
        approximately line up sensor data in time.
        """
        cam_subs = self.image_subscribers()
        laser_subs = self.scan_subs()

        self.topics = [sub.name for sub in cam_subs + laser_subs]

        self.approx_ts = ApproximateTimeSynchronizer(cam_subs + laser_subs, 10, slop=0.51)
        self.approx_ts.registerCallback(self.cb)
        rospy.spin()

    def cb(self, *subscribers):
        """
        The callback invoked upon receiving sensor data.
        `subscribers` is a list of messages received from the sensors.
        In order, subscribers contains messages for:
            FRONT_CAMERA
            BACK_CAMERA
            LEFT_CAMERA
            RIGHT_CAMERA
            FRONT_LEFT_LASER
            FRONT_RIGHT_LASER
            CENTER_LASER
        """
        front_camera, back_camera, left_camera, right_camera = subscribers[:4]
        frontleft_laser, frontright_laser, center_laser = subscribers[4:]

        frontleft_cloud = self.laser_projector.projectLaser(frontleft_laser)
        frontright_cloud = self.laser_projector.projectLaser(frontright_laser)
        center_cloud = self.laser_projector.projectLaser(center_laser)

        self.msg_data[SensorSource.FRONT_CAMERA] = front_camera
        self.msg_data[SensorSource.BACK_CAMERA] = back_camera
        self.msg_data[SensorSource.LEFT_CAMERA] = left_camera
        self.msg_data[SensorSource.RIGHT_CAMERA] = right_camera

        self.msg_data[SensorSource.FRONT_LEFT_LASER] = frontleft_cloud
        self.msg_data[SensorSource.FRONT_RIGHT_LASER] = frontright_cloud
        self.msg_data[SensorSource.CENTER_LASER] = center_cloud
        self.loaded_data = True


if __name__ == "__main__":
    rospy.init_node("estimator_node", anonymous=True, disable_signals=True)
    scene_parser = SceneParser(30, verbose=True)

    # collect some wrenches and 0plot:: we are not doing any wrenches anymore, no?
    try:
        while not rospy.is_shutdown():
            scene_parser.subscribers()
    except rospy.ROSInterruptException:
        rospy.logfatal("shutting down ros")