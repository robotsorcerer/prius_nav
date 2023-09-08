#!/usr/bin/env python3

import argparse
import datetime
import os
import cv2
import rospy
import pickle
import numpy as np
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from prius_msgs.msg import Control

from collections import deque

import rospkg
rospack = rospkg.RosPack()
from os.path import join
from collections import deque
from operator import itemgetter
from typing import List, Tuple, Optional

from scene_parser import SceneParser, SensorSource, SensorCollection, ros_camera_intrinsics

CAMERA_SOURCES = [
    key for key in SensorSource
    if (key in SensorSource.ALL_CAMERAS) and (not key.name.startswith("ALL"))
]

LASER_SOURCES = [
    SensorSource.CENTER_LASER,
    SensorSource.FRONT_LEFT_LASER,
    SensorSource.FRONT_RIGHT_LASER
]

OTHER_SOURCES = [
    SensorSource.ODOMETRY,
    SensorSource.CONTROLS
]

ALL_SENSORS = CAMERA_SOURCES + LASER_SOURCES + OTHER_SOURCES
ALL_KEYS = (list(map(lambda x: x.name, CAMERA_SOURCES))
            + list(map(lambda x: x.name, LASER_SOURCES)) + ['position', 'orientation'])

data_dir = rospack.get_path('exploratorio') + '/data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

def image_to_numpy(img: Image) -> np.ndarray:
    return np.frombuffer(img.data, dtype=np.uint8)

class SimpleDatasetBuilder(SceneParser):
    def __init__(
        self,
        capacity,
        rate,
        persist_period=-1,
        verbose=True,
        dataset_filepath=None,
        **kwargs
    ):
        super().__init__(rate, verbose=verbose, **kwargs)
        self._last_write_time = rospy.Time.now()
        self.capacity = capacity
        self.data = {src: deque(maxlen=capacity) for src in ALL_KEYS}
        self.tag = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        if dataset_filepath is None:
            dataset_filepath = f"data_dump_{self.tag}.p"

        self.dataset_filepath = dataset_filepath
        self._abs_dataset_path = os.path.abspath(self.dataset_filepath)
        self.persist_period = persist_period

    def cb(self, *subscribers):
        """
        Collects data from sensors and stores data from camera sensors
        in the replay buffer.

        Periodically writes replay buffer to disk according to the `self.persist_period`.
        """
        super().cb(*subscribers)
        for camera in CAMERA_SOURCES:
            image = self.msg_data[camera]
            data = image_to_numpy(image)
            self.data[camera.name].append(data)

        odo_data = self.msg_data[SensorSource.ODOMETRY]
        odo_pos = np.array([odo_data.position.x, odo_data.position.y, odo_data.position.z])
        odo_ori = odo_data.orientation
        odo_ori = np.array([odo_ori.x, odo_ori.y, odo_ori.z, odo_ori.w])

        self.data['position'].append(odo_pos)
        self.data['orientation'].append(odo_ori)

        delta = rospy.Time.now() - self._last_write_time
        if (self.persist_period > 0) and (delta.to_sec() >= self.persist_period):
            with open(self.dataset_filepath, "wb") as f:
                pickle.dump(self.data, f)
                if self.verbose:
                    rl = len(self.data[ALL_KEYS[0]])
                    abspath = self._abs_dataset_path
                    rospy.loginfo(f"Wrote replay buffer of size {rl} to {abspath}")
                self._last_write_time = rospy.Time.now()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compress", action='store_true')
    parser.add_argument("--capacity", type=int, default=30_000)
    args = parser.parse_args()
    rospy.init_node("data_dumper", anonymous=True, disable_signals=True)
    scene_parser = SimpleDatasetBuilder(
        args.capacity,
        30,
        persist_period=30,
        verbose=True,
        compressed_imgs=args.compress
    )

    try:
        while not rospy.is_shutdown():
            scene_parser.subscribers()
    except rospy.ROSInterruptException:
        rospy.logfatal("shutting down ros")
