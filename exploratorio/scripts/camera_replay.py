#!/usr/bin/env python3

import numpy as np
import rospy

from collections import deque
from operator import itemgetter
from typing import List

from sensor_msgs.msg import Image

from scene_parser import SceneParser, SensorSource

CAMERA_SOURCES = [
    key for key in SensorSource
    if (key in SensorSource.ALL_CAMERAS) and (not key.name.startswith("ALL"))
]

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

    def sample(self, batch_size: int, horizon: int, sensors: SensorSource) -> dict:
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
                batches[key].append(data)
        return batches

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

    def __init__(self, replay_buffer: CameraSequenceReplayBuffer, rate, **kwargs):
        super().__init__(rate, **kwargs)
        self.replay_buffer = replay_buffer

    def cb(self, *subscribers):
        super().cb(*subscribers)
        for camera in CAMERA_SOURCES:
            image = self.msg_data[camera]
            data = image_to_numpy(image)
            self.replay_buffer.store(data, camera)


if __name__ == "__main__":
    rospy.init_node("estimator_node", anonymous=True, disable_signals=True)
    buffer = CameraSequenceReplayBuffer(5)
    scene_parser = CameraReplay(buffer, 30, verbose=True)

    # collect some wrenches and 0plot:: we are not doing any wrenches anymore, no?
    try:
        while not rospy.is_shutdown():
            scene_parser.subscribers()
    except rospy.ROSInterruptException:
        rospy.logfatal("shutting down ros")
