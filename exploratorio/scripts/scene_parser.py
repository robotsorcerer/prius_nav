#!/usr/bin/env python3

import os
import sys
import rospy
import pickle
import logging
import numpy as np
from os.path import join 
from dataclasses import dataclass
import laser_geometry.laser_geometry as lg
from sensor_msgs.msg import CameraInfo, Image, CompressedImage, LaserScan, PointCloud2
from message_filters import Subscriber, ApproximateTimeSynchronizer
from typing import List, Optional, FrozenSet, Set

import rospkg
rospack = rospkg.RosPack()
np.float = float

from enum import IntFlag, auto

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOGGER = logging.getLogger(__name__)


def ros_camera_intrinsics(msg: CameraInfo) -> np.ndarray:
    return np.array(msg.K).reshape(3, 3)

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

@dataclass
class SensorCollection:
    mask: SensorSource
    sensors: Set[SensorSource]

    def __len__(self):
        return len(self.sensors)

    @staticmethod
    def from_mask(sensors: SensorSource):
        sensor_set = set()
        for key in SensorSource:
            if key in sensors:
                sensor_set.add(key)
        return SensorCollection(sensors, sensor_set)

    @staticmethod
    def from_set(sensor_set: Set[SensorSource]):
        mask = 0
        for sensor in sensor_set:
            mask |= sensor
        return SensorCollection(mask, sensor_set)

    @staticmethod
    def empty():
        return SensorCollection(SensorSource.ALL_SENSORS & 0, set())

class SceneParser():
    """
    This ROS node collects data from various sensors in the prius robot.

    It subscribes to /prius/{front,back,left,right}_camera/image_raw to
    collect image data, and subscribes to /prius/{front_left,front_right,center}_laser/scan
    for laser scan data. Upon receiving data from these topics, the data is
    stored in `msg_data[sensor_enum]` where `sensor_enum` is `SensorSource` value
    corresponding to the relevant sensor.
    """

    def __init__(
        self,
        rate,
        compressed_imgs=True,
        verbose=False,
        data_collect=False,
        sensor_info_path: Optional[str] = 'sensor_info.p',
    ):
        """
        rate: integer, not currently used
        """
        super(SceneParser, self).__init__()
        self.first_log = True  # for debugging
        self.loaded_data = False

        self.verbose = verbose
        self.basetopic = "/prius"
        self.suffix_cam_topic = "_camera/image_raw/compressed"
        self.suffix_laser_topic = "_laser/scan"
        self.topics = []

        self.msg_data = {key: None for key in SensorSource if not key.name.startswith("ALL")}
        self.sensor_info = {key: None for key in SensorSource if not key.name.startswith("ALL")}
        self.compressed_imgs = compressed_imgs

        data_dir = rospack.get_path('exploratorio') + '/data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        self.sensor_info_path = join(data_dir, sensor_info_path)

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
        img_topic_msg_type = CompressedImage if self.compressed_imgs else Image
        return [
            Subscriber(rf"{self.basetopic}/{loc}{self.suffix_cam_topic}", img_topic_msg_type)
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

    def poll_sensor_info(self):
        """
        Polls for sensor information.
        Currently, this is used to retrieve data from the `camera_info` topics,
        for the purpose of storing some metadata from the cameras
        (e.g., intrinsics matrices, image dimensions, etc).
        Since this data stays constant, we do not need to subscribe to these topics,
        we can just retrieve the data once.
        """
        for key in SensorSource:
            if (key in SensorSource.ALL_CAMERAS) and (not key.name.startswith("ALL")):
                topic = f"{self.basetopic}/{key.name.lower()}/camera_info"
                if self.verbose:
                    rospy.loginfo(f"Waiting for camera info from {topic}...")
                msg = rospy.wait_for_message(topic, CameraInfo)
                if self.verbose:
                    rospy.loginfo(f"Intrinsics: {msg.K}")
                    rospy.loginfo("Done.")
                self.sensor_info[key] = msg

        if self.sensor_info_path is not None:
            with open(self.sensor_info_path, "wb") as f:
                pickle.dump(self.sensor_info, f)
                if self.verbose:
                    rospy.loginfo(f"Wrote sensor info to {os.path.abspath(self.sensor_info_path)}")

    def subscribers(self):
        """
        Instantiates the Subscribers for the topics of interest, registers
        the callback, and spins ROS.

        Since we are collecting data from a multitude of sensors, we collect all
        subscribers in a ApproximateTimeSynchronizer message filter, so we will
        approximately line up sensor data in time.

        Before subscribing to these topics, this method polls for messages from the cameras
        to retrieve the camera intrinsics matrices. We do not need to subscribe to these
        topics, because the intrinsics matrices will remain fixed.
        """
        self.poll_sensor_info()
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

    try:
        while not rospy.is_shutdown():
            scene_parser.subscribers()
    except rospy.ROSInterruptException:
        rospy.logfatal("shutting down ros")
