#!/usr/bin/env python3

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


def image_to_numpy(img: Image) -> np.ndarray:
    return np.frombuffer(img.data, dtype=np.uint8).astype(np.float32) / 255.0

class OnlineDataset:
    def __init__(
        self,
        name: str,
        capacity: int,
        *topics: Tuple[Tuple[str, Any]],
        queue_length=10,
        slop=0.51,
        **kwargs
    ):
        self.name = name
        self.buf = deque(maxlen=capacity)
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
            self.buf.append(data)
            rospy.loginfo(f"Stored data, current buffer size is {len(self)}")

    def sample_data(self, msg):
        batch_indices = np.random.randint(len(self) - msg.horizon + 1, size=msg.batch_size)
        breakpoint()
        batches = [list(zip(*self.buf[batch_indices + i])) for i in range(msg.horizon)]
        self.query_pub.publish(UInt8MultiArray(data=batches))

    def __len__(self):
        return len(self.buf)

class OnlineCameraDataset(OnlineDataset):
    def __init__(self, name: str, capacity: int, *topics, **kwargs):
        super().__init__(name, capacity, *topics, **kwargs)

    def process_data(self, *msgs):
        return [image_to_numpy(msg) for msg in msgs]

    def should_store_data(self, data: List[np.ndarray]) -> bool:
        if len(self) == 0:
            return True
        last_data = self.buf[-1]
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
