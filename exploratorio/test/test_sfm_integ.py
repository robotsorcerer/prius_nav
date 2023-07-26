import os
import pickle
import rospy
import threading
import time
import torch

from scene_parser import SensorSource
from camera_replay import CameraReplay, CameraSequenceReplayBuffer
from sfm.sfm_model import SfmLearner

BATCH_SIZE = 5
HORIZON = 3

rospy.init_node("test_sfm_integ_sandbox", anonymous=True, disable_signals=True)
buf = CameraSequenceReplayBuffer(500)
node = CameraReplay(buf, 30, persist_period=-1, verbose=True)

thr = threading.Thread(target=node.subscribers, args=(), kwargs={})
if not rospy.is_shutdown():
    thr.start()

intrinsics = None
while intrinsics is None:
    intrinsics = node.camera_intrinsics(SensorSource.FRONT_CAMERA)

if os.path.exists("still-dataset.p"):
    with open("still-dataset.p", "rb") as f:
        dataset = pickle.load(f)
else:
    dataset = node.replay_buffer
    rospy.loginfo("Waiting a bit for data to be collected...")
    time.sleep(10)
    rospy.loginfo(f"Buffer now has {len(dataset)} datapoints")

batch = dataset.sample(
    BATCH_SIZE,
    HORIZON,
    SensorSource.FRONT_CAMERA
)[SensorSource.FRONT_CAMERA]

batch = [torch.tensor(img, requires_grad=False) / 255.0 for img in batch]

model = SfmLearner(intrinsics)

batch = [t.permute(0, 3, 1, 2).to(model.device) for t in batch]

loss, depths, pose, exps = model.train_step(batch, return_outputs=True)

rospy.loginfo("Computed SfM loss and gradients for batch")
rospy.loginfo(f"SfM loss: {loss.item()}")
rospy.loginfo(f"Computed multi-scale depth maps with shapes {[d.shape for d in depths]}")
rospy.loginfo(f"Computed pose with shape {pose.shape}")
