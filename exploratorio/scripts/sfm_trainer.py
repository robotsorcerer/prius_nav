#!/usr/bin/env python3

import argparse
import logging
import numpy as np
import os
import pickle
import rospy
import threading
import time
import torch

# For experiment tracking
from aim import Run, Image
from typing import List

from scene_parser import SensorSource, SensorCollection, ros_camera_intrinsics
from camera_replay import CameraReplay, CameraSequenceReplayBuffer
from sfm.sfm_model import SfmLearner

logging.basicConfig(level=logging.INFO)

def process_img_data(img) -> torch.Tensor:
    return torch.tensor(img, requires_grad=False).permute(0, 3, 1, 2) / 255.0

class SfMTrainer:
    def __init__(
        self,
        run: Run,
        batch_size=32,
        horizon=3,
        demo_every=50,
        logdir='results',
        model_params=dict(),
    ):
        self.run = run
        self.batch_size = batch_size
        self.horizon = horizon
        self.logdir = logdir
        self.model = None
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.update_step = 0
        self.demo_every = demo_every
        self.model_params = model_params

    def init_experiment(self):
        if self.model is None:
            self.model = SfmLearner(self.intrinsics, **self.model_params)
            if isinstance(self.run, Run):
                self.run['hparams'] = self.hparams_dict()

    def hparams_dict(self):
        return {
            'model': self.model.hparams,
            'batch_size': self.batch_size,
            'horizon': self.horizon,
        }

    @property
    def intrinsics(self) -> torch.Tensor:
        raise NotImplementedError

    def available_sensors(self) -> SensorCollection:
        raise NotImplementedError

    @property
    def buffer(self):
        raise NotImplementedError

    def sample_batch(self, batch_size=None, horizon=None) -> List[torch.Tensor]:
        sensors = self.available_sensors()
        batch_size = batch_size or (self.batch_size // len(sensors))
        batch_data = self.buffer.sample(batch_size, horizon or self.horizon, sensors.mask)
        device = (self.model and self.model.device) or self.device
        for sensor in sensors.sensors:
            """
            batch_data: { sensor1: HORIZON x (B, 3, W, H), sensor2: HORIZON x (B, 3, W, H), ... }
            """
            batch_data[sensor] = [process_img_data(img).to(device) for img in batch_data[sensor]]

        """
        We concat data from different sensors into one batch, producing
        HORIZON x (len(sensors) * B, 3, W, H)
        """
        return list(map(torch.concat, zip(*batch_data.values())))

    def train_step(self):
        self.update_step += 1
        batch = self.sample_batch()
        return self.model.train_step(batch)

    def train(self, num_steps):
        step = 0
        while True:
            step += 1

            start = rospy.Time.now()
            artifacts = self.train_step()
            end = rospy.Time.now()
            batch_update_time = (end - start).to_sec()

            if self.run is not None:
                artifacts.track(self.run)
                self.run.track(batch_update_time, name='batch_update_time')

            if step % self.demo_every == 0:
                batch = self.sample_batch(batch_size=1)
                random_idx = np.random.randint(batch[0].shape[0])
                tgt_img = batch[0][random_idx, :, :, :]
                disp = self.model.disp_net(tgt_img.unsqueeze(0))[0].squeeze()
                depth = 1 / disp
                depth = torch.stack([depth] * 3)
                depth_demo = torch.concat([tgt_img[[2, 1, 0], :, :], depth], axis=-1)
                if self.run is not None:
                    self.run.track(Image(depth_demo), name='depth')
            if (num_steps > 0) and (step >= num_steps):
                break

class OnlineSfMTrainer(SfMTrainer):
    def __init__(
        self,
        run: Run,
        buffer_size=10_000,
        rate=30,
        persist_period=-1,
        node_name='online_sfm_learner',
        buffer_verbose=True,
        **kwargs
    ):
        super().__init__(run, **kwargs)
        self.node_name = node_name

        rospy.init_node(node_name, anonymous=True, disable_signals=True)
        buf = CameraSequenceReplayBuffer(buffer_size)
        self.node = CameraReplay(buf, rate, persist_period=persist_period, verbose=buffer_verbose)

        self.thr = threading.Thread(target=self.node.subscribers, args=(), kwargs={})
        if not rospy.is_shutdown():
            self.thr.start()

        intrinsics = None
        while intrinsics is None:
            """
            Camera intrinsics are the same for each camera in this case.
            Can be verified with

            ```bash
            rostopic echo /prius/[front|back|left|right]_camera/camera_info
            ```
            """
            intrinsics = self.node.camera_intrinsics(SensorSource.FRONT_CAMERA)
        self._intrinsics = intrinsics

    @property
    def intrinsics(self):
        return self._intrinsics

    @property
    def buffer(self):
        return self.node.replay_buffer

    def hparams_dict(self):
        hparams = super().hparams_dict()
        hparams['training_format'] = 'online'
        return hparams

    def available_sensors(self) -> SensorCollection:
        node = self.node
        i = 0
        available_sensors = SensorCollection.empty()
        while len(available_sensors) == 0:
            available_sensors = node.replay_buffer.available_sensors(
                self.horizon,
                SensorSource.ALL_CAMERAS
            )
            if len(available_sensors) != 0:
                break
            rospy.logwarn(f"Not enough data to sample batch. Retrying in {i} second(s).")
            time.sleep(2 ** min(i, 5))
            i += 1
        return available_sensors


class OfflineSfMTrainer(SfMTrainer):
    def __init__(
        self,
        run: Run,
        dataset_path: str = "dataset.p",
        sensor_info_path: str = "sensor_info.p",
        buffer_verbose: bool = True,
        **kwargs
    ):
        super().__init__(run, **kwargs)

        with open(dataset_path, "rb") as f:
            self._buffer = pickle.load(f)

        with open(sensor_info_path, "rb") as f:
            self.sensor_info = pickle.load(f)

        self._available_sensors = self.buffer.available_sensors(
            self.horizon,
            SensorSource.ALL_CAMERAS
        )

        sensor_keys = list(self.sensor_info.keys())
        intrinsics = ros_camera_intrinsics(self.sensor_info[sensor_keys[0]])
        self._intrinsics = intrinsics

    @property
    def intrinsics(self):
        return self._intrinsics

    @property
    def buffer(self):
        return self._buffer

    def hparams_dict(self):
        hparams = super().hparams_dict()
        hparams['training_format'] = 'offline'
        return hparams

    def available_sensors(self) -> SensorCollection:
        return self._available_sensors

def main(args):
    run = None if args.dryrun else Run(experiment=args.experiment_name)

    model_params = {
        'mask_loss_weight': args.mask_loss_weight,
        'photo_loss_weight': args.photo_loss_weight,
        'smooth_loss_weight': args.smooth_loss_weight,
        'lr': args.lr,
    }

    if args.offline:
        rospy.init_node("offline_sfm_trainer")
        trainer = OfflineSfMTrainer(
            run,
            batch_size=args.batch_size,
            dataset_path=args.data_path,
            sensor_info_path=args.sensor_path,
            demo_every=args.demo_every,
            model_params=model_params,
        )
    else:
        trainer = OnlineSfMTrainer(
            run,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            persist_period=args.persist_period,
            demo_every=args.demo_every,
            model_params=model_params,
        )
    batch = trainer.sample_batch()
    rospy.loginfo(f"Length of batch sequence: {len(batch)}")
    rospy.loginfo(f"Shape of batches: {batch[0].shape}")

    trainer.init_experiment()
    trainer.train(args.num_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Do not use offline data"
    )
    parser.add_argument("--experiment-name", type=str, default="sfm")
    parser.add_argument("--data-path", type=str, default="dataset.p")
    parser.add_argument("--sensor-path", type=str, default="sensor_info.p")
    parser.add_argument("--buffer-size", type=int, default=10_000)
    parser.add_argument("--persist-period", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mask-loss-weight", type=float, default=1.0)
    parser.add_argument("--photo-loss-weight", type=float, default=1.0)
    parser.add_argument("--smooth-loss-weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-steps", type=int, default=1_000)
    parser.add_argument("--demo-every", type=int, default=50, help="Steps between image logs")
    parser.add_argument("--dryrun", action="store_true", help="Do not track experiment")

    args = parser.parse_args()
    main(args)
