#!/usr/bin/env python3

import argparse
import numpy as np
import pickle
import rospy

from scene_parser import SensorSource
from camera_replay import CameraSequenceReplayBuffer

def main(args):
    with open(args.data_path, 'rb') as f:
        buffer = pickle.load(f)

    keys = buffer.bufs.keys()

    data = dict()

    if args.max_images <= 0:
        for key in keys:
            data[key.name] = buffer.bufs[key]
    else:
        if args.localized:
            batch = buffer.sample(1, args.max_images, SensorSource.ALL_CAMERAS, decode=True)
            for key in batch.keys():
                images = np.stack(batch[key]).squeeze()
                data[key.name] = images
        else:
            batch = buffer.sample(args.max_images, 1, SensorSource.ALL_CAMERAS, decode=True)
            for key in batch.keys():
                data[key.name] = batch[key][0]


    with open(args.out_path, 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="dataset.p")
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--out-path", type=str, default="sample-data.p")
    parser.add_argument("--localized", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
