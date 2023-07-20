#!/usr/bin/env python3

from scene_parser import SceneParser

import rospy
import threading
import time


def print_camera_data(scene_parser: SceneParser, key: str):
    data = scene_parser.msg_data[key]
    print(key)
    print('-' * len(key))
    print("Header:")
    print(data.header)
    print(f"Data length: {len(data.data)}")
    print('-' * len(key) + "\n")


def print_laser_data(scene_parser: SceneParser, key: str):
    data = scene_parser.msg_data[key]
    print(key)
    print('-' * len(key))
    print("Header:")
    print(data.header)
    print(f"Data length: {len(data.data)}")
    print('-' * len(key))


if __name__ == "__main__":
    print("Running")
    rospy.init_node("estimator_node", anonymous=True, disable_signals=True)
    scene_parser = SceneParser(30, verbose=True)
    print(scene_parser.msg_data)

    # collect some wrenches and 0plot
    try:
        thr = threading.Thread(target=scene_parser.subscribers, args=(), kwargs={})
        if not rospy.is_shutdown():
            thr.start()
        while not rospy.is_shutdown():
            if scene_parser.loaded_data:
                print("CAMERA DATA")
                print('=' * len("CAMERA DATA"))
                for camera_key in ['front', 'back', 'left', 'right']:
                    print_camera_data(scene_parser, f"{camera_key}_camera")
                print('=' * len("CAMERA DATA") + "\n")
                print("LASER DATA")
                print('=' * len("LASER DATA"))
                for laser_key in ['front_left', 'front_right', 'center']:
                    print_laser_data(scene_parser, f"{laser_key}_laser")
                print('=' * len("LASER DATA") + "\n")
            time.sleep(2)
    except rospy.ROSInterruptException:
        rospy.logfatal("shutting down ros")
    except KeyboardInterrupt:
        thr.join(1)
