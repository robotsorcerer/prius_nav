#!/usr/bin/env python3

import argparse
import numpy as np
import os
import pickle
import rospy
import secrets

from PIL import Image

from camera_replay import CameraSequenceReplayBuffer
from scene_parser import SensorSource

def main(args):
    name = args.name if args.name is not None else f"car_{secrets.token_urlsafe(5)}"
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    rospy.loginfo("Loading dataset")
    with open(args.data_path, "rb") as f:
        buffer = pickle.load(f)
    rospy.loginfo("Loaded dataset")
    for i in range(args.n):
        rospy.loginfo(f"Batch {i+1}/{args.n}")
        if buffer.can_sample_sequence(args.horizon, SensorSource.ALL_CAMERAS):
            sampled_data = buffer.sample(
                1,
                args.horizon,
                SensorSource.ALL_CAMERAS
            )
            for key in sampled_data.keys():
                data = np.array(sampled_data[key]).squeeze()
                for t in range(args.horizon):
                    img_data = data[t, :, :, ::-1]
                    img = Image.fromarray(img_data)
                    path = f"{args.out_path}/{name}-{key.name}_batch{i}_step{t}.png"
                    img.save(path)
                    rospy.loginfo(f"Wrote image to {path}")


if __name__ == "__main__":
    rospy.init_node("dummy")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="take3.p")
    parser.add_argument("--out-path", type=str, default="/tmp/dump")
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    main(args)
