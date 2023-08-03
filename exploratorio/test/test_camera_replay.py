#!/usr/bin/env python3

from scene_parser import SensorSource
from camera_replay import CameraReplay, CameraSequenceReplayBuffer

import rospy
import threading
import time
import numpy as np

CAPACITY = 1000
HORIZON = 3
BATCH_SIZE = 5

CAMERA_SENSORS = [SensorSource.FRONT_CAMERA,
                  SensorSource.BACK_CAMERA,
                  SensorSource.LEFT_CAMERA,
                  SensorSource.RIGHT_CAMERA]

SEPARATOR = ">>" + '=' * 20 + ">>"

if __name__ == "__main__":
    rospy.init_node("estimator_node", anonymous=True, disable_signals=True)
    buffer = CameraSequenceReplayBuffer(CAPACITY)
    buffer_node = CameraReplay(buffer, 30)

    try:
        cb_thread = threading.Thread(
            target=lambda: buffer_node.subscribers()
        )
        cb_thread.daemon = True
        cb_thread.start()


        # wait for a few secs after thread has spawned before starting identification
        # time.sleep(12)

        while not rospy.is_shutdown():
            rospy.loginfo(f"Replay buffer size: {len(buffer_node.replay_buffer)}/{CAPACITY}")
            buffer_sizes = {key: len(buffer_node.replay_buffer[key]) for key in CAMERA_SENSORS}
            rospy.loginfo(f"Buffers' length: {[(k, v) for (k,v) in buffer_sizes.items()]}")
            rospy.loginfo(f"Sampling batch of {BATCH_SIZE} sequences of length {HORIZON} from each sensor")
            if buffer_node.replay_buffer.can_sample_sequence(HORIZON, SensorSource.ALL_CAMERAS):
                sampled_data = buffer_node.replay_buffer.sample(
                    BATCH_SIZE,
                    HORIZON,
                    SensorSource.ALL_CAMERAS
                )
                for key in sampled_data.keys():
                    data = np.array(sampled_data[key])
                    rospy.loginfo(f"\t{key.name}: Received data of size {data.shape}")
                    rospy.loginfo(f"\t\t{data}")
            else:
                rospy.logwarn("Not enough data to sample sequence from all sensors")
            rospy.loginfo(SEPARATOR)
            time.sleep(1)
    except rospy.ROSInterruptException:
        rospy.logfatal("shutting down ros")