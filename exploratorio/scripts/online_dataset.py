#!/usr/bin/env python3

# DEPRECATED: not using this anymore

import logging
import numpy as np
import rospy
import sensor_msgs

from collections import deque
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8MultiArray
from prius_msgs.msg import DataBatchRequest
from typing import List, Tuple, Any


def image_to_numpy(img: Image, discrete=True) -> np.ndarray:
    if discrete:
        return np.frombuffer(img.data, dtype=np.uint8)
    else:
        return np.frombuffer(img.data, dtype=np.uint8).astype(np.float32) / 255.0

class SequenceReplayBuffer:
    def __init__(self, num_bufs: int, capacity: int):
        self.capacity = capacity
        self.bufs = [[] for _ in range(num_bufs)]
        self.num_bufs = num_bufs

        self._size = 0
        self._idx = 0

    def __len__(self):
        return self._size

    @property
    def full(self):
        return self._size >= self.capacity

    @property
    def last(self):
        return [buf[-1] for buf in self.bufs]

    def store(self, *data):
        if not (len(data) == self.num_bufs):
            raise ValueError(
                f"""
                Not enough data passed to buffer:
                Expected {self.num_bufs} data sources, got {len(data)}.
                """
            )
        if self._size < self.capacity:
            for (buf, datapoint) in zip(self.bufs, data):
                buf.append(datapoint)
        else:
            for (buf, datapoint) in zip(self.bufs, data):
                buf[self._idx] = datapoint
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample_batch(self, batch_size, horizon):
        if horizon > self._size:
            rospy.logwarn("Trying to sample oversized sequence from buffer")
            return None
        batch_indices_raw = np.random.randint(len(self) - horizon + 1, size=batch_size)
        batches = [[] for _ in range(horizon)]
        for k in range(horizon):
            for batch_index_raw in batch_indices_raw:
                batch_index = (batch_index_raw + self._idx * self.full + k) % self.capacity
                for buf in self.bufs:
                    batches[k].append(buf[batch_index])
        return batches

    def sample_batch_opt(self, batch_size, horizon):
        if horizon > self._size:
            rospy.logwarn("Trying to sample oversized sequence from buffer")
            return None
        batch_indices_raw = np.random.randint(len(self) - horizon + 1, size=batch_size)
        batches = []
        for i in range(horizon):
            batch_indices = (batch_indices_raw + self._idx * self.full + i) % self.capacity
            breakpoint()
            batches.append([np.take(buf, batch_indices) for buf in self.bufs])
        return batches

class OnlineDataset:
    def __init__(
        self,
        name: str,
        capacity: int,
        *topics: Tuple[Tuple[str, Any]],
        rate=-1,
        queue_length=10,
        slop=0.51,
        **kwargs
    ):
        self.name = name
        self.buf = SequenceReplayBuffer(len(topics), capacity)
        self.topic_names = [topic[0] for topic in topics]
        self.topic_types = [topic[1] for topic in topics]
        self.data_subs = [Subscriber(*topic) for topic in list(topics)]
        for sub in self.data_subs:
            print(f"Name: {sub.name}, class: {sub.data_class}, callback: {sub.callback_args}")
        self.synchronizer = ApproximateTimeSynchronizer(self.data_subs, queue_length, slop=slop)
        self.synchronizer.registerCallback(self.store_data)
        self.query_sub_topic = f"{self.name}/sample_data"
        self.query_pub_topic = f"{self.name}/data_batch"
        self.query_sub = Subscriber(self.query_sub_topic, DataBatchRequest)
        self.query_sub.registerCallback(self.sample_data)
        self.query_pub = rospy.Publisher(
            self.query_pub_topic,
            UInt8MultiArray,
            queue_size=queue_length
        )
        self.rate = rate
        self._last_store_time = None

    def process_data(self, *msgs):
        """
        Process the ROS msgs and return data to store in the buffer
        """
        return msgs

    def should_store_data(self, data) -> bool:
        """
        Decides whether new batch of data should be stored in the buffer.
        For instance, if the data is redundant (e.g., it is exactly the same as the
        data that was last persisted), then this should return False.
        """
        return True

    def store_data(self, *msgs):
        data = self.process_data(*msgs)
        if self.should_store_data(data):
            self.buf.store(*data)
            rospy.loginfo(f"Stored data, current buffer size is {len(self)}")

    def sample_data(self, msg):
        batches = self.buf.sample_batch(msg.batch_size, msg.horizon)
        self.query_pub.publish(UInt8MultiArray(data=batches))

    def __len__(self):
        return len(self.buf)

class OnlineCameraDataset(OnlineDataset):
    def __init__(self, name: str, capacity: int, *topics, **kwargs):
        super().__init__(name, capacity, *topics, **kwargs)

    def process_data(self, *msgs):
        return [image_to_numpy(msg, discrete=True) for msg in msgs]

    def should_store_data(self, data: List[np.ndarray]) -> bool:
        if len(self) == 0:
            return True
        last_data = self.buf.last
        for (data_new, data_old) in zip(data, last_data):
            # DEBUG
            return False
            if not np.all(np.isclose(data_new, data_old)):
                return True
        return False


if __name__ == "__main__":
    rospy.init_node("data_hosting_node", anonymous=True, disable_signals=True)
    dataset = OnlineCameraDataset(
        "camera_dataset",
        1000,
        ("/prius/front_camera/image_raw", Image),
        ("/prius/back_camera/image_raw", Image),
        ("/prius/left_camera/image_raw", Image),
        ("/prius/right_camera/image_raw", Image)
    )

    rospy.spin()
