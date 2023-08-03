#!/usr/bin/env python3

import os
import cv2
import rospy
import pickle
import numpy as np
from sensor_msgs.msg import Image

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


data_dir = rospack.get_path('exploratorio') + '/data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

def image_to_numpy(img: Image) -> np.ndarray:
    return np.frombuffer(img.data, dtype=np.uint8)

class CameraSequenceReplayBuffer:
    """
    Dynamics buffer to store data from multiple cameras, allowing for
    batches of contiguous (in time) images to be sampled.
    """

    def __init__(self, capacity):
        """
        capacity: int
            The total capacity (in number of images) across all image sensor data.
        """
        self.capacity = capacity
        self.bufs = {key: [] for key in CAMERA_SOURCES}

    def __len__(self):
        return sum([len(self.bufs[k]) for k in self.bufs.keys()])

    def __getitem__(self, key):
        return self.bufs[key]

    def store(self, img: List, sensor: SensorSource):
        """
        Stores image data in the buffer corresponding to the sensor specified by `sensor`.
        """
        self.bufs[sensor].append(img)
        self.evict()

    def evict(self):
        """
        Evicts data to prevent the buffer from exceeding its capacity.
        In this implementation, we allocate an even amount of capacity for
        each sensor buffer, and evict the oldest data in a given sensor buffer
        when that buffer has exceeded its capacity.
        """
        keys = self.bufs.keys()
        N = len(keys)
        for k in keys:
            if len(self.bufs[k]) >= self.capacity / N:
                self.bufs[k].pop(0)

    def sample(self, batch_size: int, horizon: int, sensors: SensorSource, decode=True) -> dict:
        """
        Samples `batch_size` batches of image sequences of length `horizon`
        from the buffers specified by `sensors`.
        For example, if `sensors` is `SensorSource.ALL_CAMERAS`, this function
        returns a dictionary containing a key for each camera source in `SensorSource`
        pointing to a batch of images for each timestep in the sequence.

        Explicitly, calling `self.sample(batch_size, 3, SensorSource.ALL_CAMERAS)` produces

        ```
        {
            SensorSource.FRONT_CAMERA: [batch_t0, batch_t1, batch_t2],
            SensorSource.BACK_CAMERA: [batch_t0, batch_t1, batch_t2],
            ...
        }
        ```

        where `batch_tk` is a batch of `batch_size` flattened images, corresponding to the
        images in the kth timestep of the sampled contiguous sequences.

        When `decode=True`, the images are all decoded by OpenCV, and instead of flat vectors,
        the `batch_tk` objects will be ndarrays with shape (batch_size, H, W, 3), where
        W and H are the image width and height.
        """
        keys = [key for key in CAMERA_SOURCES if key in sensors]
        batches = {key: [] for key in keys}
        for key in keys:
            batch_indices = np.random.randint(
                len(self.bufs[key]) - horizon + 1,
                size=batch_size
            )
            for k in range(horizon):
                shifted_indices = batch_indices + k
                data = itemgetter(*shifted_indices)(self.bufs[key])
                if batch_size == 1:
                    data = [data]
                if decode:
                    data = np.array([
                        cv2.imdecode(image_to_numpy(d), cv2.IMREAD_UNCHANGED) for d in data
                    ])
                batches[key].append(data)
        return batches

    def can_sample_sequence(self, horizon: int, sensors: SensorSource):
        """
        Indicates whether the buffer has amassed enough data to sample
        sequences of length `horizon` for each sensor described by `sensors`.
        """
        avail = self.available_sensors(horizon, sensors)
        rospy.logdebug(f"avail: {avail} sensors: {sensors}")
        return (sensors & avail.mask) == sensors

    def available_sensors(self, horizon, sensors: SensorSource) -> SensorCollection:
        sensor_set = set()
        mask = 0
        for key in SensorSource:
            rospy.loginfo(f"key: {key}")
            if (key in sensors) and (key in CAMERA_SOURCES):
                rospy.loginfo(f"key (inner loop): {key}, {len(self.bufs[key])}")
                if len(self.bufs[key]) >= horizon:
                    mask |= key
                    sensor_set.add(key)
            rospy.logdebug(f"key: {key} sensor set: {sensor_set}")
        return SensorCollection(mask, sensor_set)

class LRUCameraSequenceReplayBuffer(CameraSequenceReplayBuffer):
    """
    A `CameraSequenceReplayBuffer` with an alternate eviction strategy that
    evicts the oldest data across all sensors.

    In this implementation, the buffers for the different sensors are not given the
    same capacity. They grow dynamically as data is stored in them individually, and
    eviction of the oldest data across all sensor buffers maintains specified capacity
    for the collection of buffers.
    """

    def __init__(self, capacity):
        super().__init__(capacity)
        self.source_trace = deque()

    def store(self, img: List, sensor: SensorSource):
        super().store(img, sensor)
        self.source_trace.append(sensor)

    def evict(self):
        while len(self) >= self.capacity:
            sensor = self.source_trace.popleft()
            self.bufs[sensor].pop()

class CameraReplay(SceneParser):
    """
    ROS node that maintains a buffer of camera data to train an unsupervised
    depth and ego-motion model from pixels (see Zhou et. al, https://arxiv.org/abs/1704.07813).
    """

    def __init__(
        self,
        replay_buffer: CameraSequenceReplayBuffer,
        rate,
        persist_period=-1,
        verbose=True,
        dataset_filepath='dataset.p',
        **kwargs
    ):
        super().__init__(rate, verbose=verbose, **kwargs)
        self.replay_buffer = replay_buffer
        self.persist_period = persist_period
        self._last_write_time = rospy.Time.now()
        self.dataset_filepath = join(data_dir, dataset_filepath)
        self._abs_dataset_path = os.path.abspath(self.dataset_filepath)

    def camera_intrinsics(self, sensor: SensorSource) -> Optional[np.ndarray]:
        """
        Retrieves the camera intrinsics for the given camera `sensor`.
        Returns `None` if the camera sensor information has not yet been retrieved.
        """
        if self.sensor_info[sensor] is None:
            return None
        return ros_camera_intrinsics(self.sensor_info[sensor])

    def camera_image_dimensions(self, sensor: SensorSource) -> Optional[Tuple[int, int]]:
        """
        Retrieves the image dimensions (width, height) for the given camera `sensor`.
        Returns `None` if the camera sensor information has not yet been retrieved.
        """
        if self.sensor_info[sensor] is None:
            return None
        info = self.sensor_info[sensor]
        return (info.width, info.height)

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
            self.replay_buffer.store(data, camera)
        delta = rospy.Time.now() - self._last_write_time
        if (self.persist_period > 0) and (delta.to_sec() >= self.persist_period):
            with open(self.dataset_filepath, "wb") as f:
                pickle.dump(self.replay_buffer, f)
                if self.verbose:
                    rl = len(self.replay_buffer)
                    abspath = self._abs_dataset_path
                    rospy.loginfo(f"Wrote replay buffer of size {rl} to {abspath}")
                self._last_write_time = rospy.Time.now()


if __name__ == "__main__":
    rospy.init_node("online_camera_replay", anonymous=True, disable_signals=True)
    buffer = CameraSequenceReplayBuffer(25000)
    scene_parser = CameraReplay(buffer, 30, persist_period=30, verbose=True)

    try:
        while not rospy.is_shutdown():
            scene_parser.subscribers()
    except rospy.ROSInterruptException:
        rospy.logfatal("shutting down ros")
