#!/usr/bin/env python3

import os
import cv2
import rospy
import open3d as o3d
import pickle
import numpy as np

import rospkg
import tf
import tf2_ros
rospack = rospkg.RosPack()
from typing import List, Tuple, Optional

# from open3d.geometry import get_rotation_matrix_from_xyz
from tf.transformations import quaternion_from_euler

from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image

from scene_parser import SceneParser, SensorSource, ros_camera_intrinsics
from scene_generator import SceneGenerator

CAMERA_SOURCES = [
    key for key in SensorSource
    if (key in SensorSource.ALL_CAMERAS) and (not key.name.startswith("ALL"))
]

def image_to_numpy(img: Image) -> np.ndarray:
    return np.frombuffer(img.data, dtype=np.uint8)

class Landmarker(SceneParser):
    """
    ROS node that detects landmarks from multiple cameras and registers frames for them.

    Params:
        downsample (float, 0.08): The voxel grid size for voxel downsampling pointclouds
        num_clusters (int, 6): The number of clusters for pointcloud segmentation
        max_seg_objects (int, 4): The maximum number of landmarks allowed to be detected in a scene
    """

    def __init__(
        self,
        rate,
        downsample: Optional[float] = 0.08,
        num_clusters: int = 6,
        max_seg_objects: int = 4,
        verbose=True,
        dataset_filepath='dataset.p',
        **kwargs
    ):
        super().__init__(rate, verbose=verbose, **kwargs)
        self.sg = SceneGenerator(
            downsample=downsample,
            num_clusters=num_clusters,
            max_seg_objects=max_seg_objects
        )
        self._last_write_time = rospy.Time.now()
        self.position = np.array([0, 0, 0])
        self.orientation = np.array([0, 0, np.pi])
        self.p0 = np.array([-12, 3, 0])  # initial position relative to base_link
        self.landmark_tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.landmark_tf_msgs = []

    @property
    def num_landmarks(self):
        return len(self.landmark_tf_msgs)

    def transform_to_base_link(self, p):
        """
        Transforms a coordinate relative to the agents frame to the base link

        Params:
            p (ndarray): The local coordinate

        Returns:
            ndarray: The coordinate in the frame of base_link
        """
        R = o3d.geometry.get_rotation_matrix_from_xyz(self.orientation)
        rotated_p = R.T @ p
        return rotated_p - self.position - self.p0

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

    def create_landmark_tf(self, loc, name, ref=None, parent=None):
        """
        Creates a tf message for a given landmark frame.

        Params:
            loc (ndarray): Location of frame relative to agent's frame
            name (str): Name of tf node
            ref (ndarray, None): The location of parent if not None, else frame will be given w.r.t base_link
            parent (str, None): The name of the parent node, or base_link when None

        Returns:
            TransformStamped: a tf node message that can be broadcasted.
        """
        trans = TransformStamped()
        trans.header.stamp = rospy.Time.now()
        if ref is None:
            trans.header.frame_id = "base_link"
            loc = self.transform_to_base_link(loc)
        else:
            loc = loc - ref
            trans.header.frame_id = parent
        trans.child_frame_id = name
        trans_t = trans.transform.translation
        trans_t.x, trans_t.y, trans_t.z = loc
        trans.transform.translation = trans_t
        trans_r = trans.transform.rotation
        trans_r.x, trans_r.y, trans_r.z, trans_r.w = quaternion_from_euler(0, 0, 0)
        trans.transform.rotation = trans_r
        self.landmark_tf_msgs.append(trans)
        return trans

    def cb(self, *subscribers):
        """
        Collects images from cameras, generates 360-degree pointcloud,
        and discovers landmarks via clustering. Local frames are associated
        to each landmark and added to the ROS tf tree.
        """
        super().cb(*subscribers)
        img_multi = {
            key.name: image_to_numpy(self.msg_data[key]).reshape(800, 800, 3)
            for key in CAMERA_SOURCES
        }

        if self.num_landmarks == 0:  # Debugging
            landmarks = self.sg.find_landmarks(img_multi)
            if len(landmarks) > 0:
                first_landmark_tf = self.create_landmark_tf(
                    landmarks[0].centroid,
                    f"landmark{self.num_landmarks}",
                    ref=None,
                    parent=None
                )
                rospy.loginfo(f"Published landmark: {first_landmark_tf.child_frame_id}")
                for (i, landmark) in enumerate(landmarks[1:]):
                    landmark_tf = self.create_landmark_tf(
                        landmark.centroid,
                        f"landmark{self.num_landmarks}",
                        ref=landmarks[i].centroid,
                        parent=f"landmark{self.num_landmarks - 1}"
                    )
                    rospy.loginfo(f"Published landmark: {landmark_tf.child_frame_id}")

            for landmark_tf in self.landmark_tf_msgs:
                trans = landmark_tf.transform.translation
                parent = landmark_tf.header.frame_id
                landmark_centroid = f"({trans.x:.2f}, {trans.y:.2f}, {trans.z:.2f})"
                rospy.loginfo(
                    f"{landmark_tf.child_frame_id} at {landmark_centroid}, parent {parent}"
                )

        for landmark_tf in self.landmark_tf_msgs:
            br = tf2_ros.StaticTransformBroadcaster()
            br.sendTransform(landmark_tf)


if __name__ == "__main__":
    rospy.init_node("landmarker", anonymous=True, disable_signals=True)
    landmarker = Landmarker(30)

    try:
        while not rospy.is_shutdown():
            landmarker.subscribers()
    except rospy.ROSInterruptException:
        rospy.logfatal("shutting down ros")
