#!/usr/bin/env python3

import argparse
import copy
import cv2
import dataclasses
import imageio
import itertools
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import open3d as o3d
import pickle
import torch

from sensor_msgs.msg import PointCloud2, PointField

from rembg import remove, new_session
from sfm.models.disp_net import DispNetS
from sklearn.cluster import KMeans
from transformers import AutoModelForDepthEstimation
from typing import List, Optional

from tqdm import tqdm

from scene_parser import SensorSource, SensorCollection, ros_camera_intrinsics

plt.rcParams['font.size'] = 20

CAMERA_LINKS = {
    'FRONT_CAMERA': {
        'pos': np.array([0.0, 1.4, 0.4]),
        'pos_level': np.array([0.0, 0.0, 0.4]),
        'angle': 0.0,
    },
    'BACK_CAMERA': {
        'pos': np.array([0.0, 1.4, -1.45]),
        'pos_level': np.array([0.0, 0.0, -1.45]),
        'angle': np.pi,
    },
    'LEFT_CAMERA': {
        'pos': np.array([1.0, 1.0, -0.7]),
        'pos_level': np.array([1.0, 0.0, -0.7]),
        'angle': np.pi / 2,
    },
    'RIGHT_CAMERA': {
        'pos': np.array([-1.0, 1.0, -0.7]),
        'pos_level': np.array([-1.0, 0.0, -0.7]),
        'angle': -np.pi / 2
    },
}

CAMERA_KEYS = list(CAMERA_LINKS.keys())

@dataclasses.dataclass
class Landmark:
    centroid: np.ndarray
    pointcloud: np.ndarray

    @property
    def size(self):
        return self.pointcloud.shape[0]


p2p = o3d.pipelines.registration.TransformationEstimationPointToPlane

def tensor_to_pointcloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Converts [N, 3] tensor representation of pointcloud to open3d PointCloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def o3d_to_ros_pointcloud(pc: o3d.geometry.PointCloud) -> PointCloud2:
    """
    Converts open3d PointCloud to a PointCloud2 ROS message
    """
    pc_data = np.asarray(pc.points, dtype=np.float32)
    pc_data *= 2
    # pc_data[:, 0] *= -1
    pc_data[:, 2] *= -1
    pc_msg = PointCloud2()
    pc_msg.header.frame_id = 'chassis'
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=8, datatype=PointField.FLOAT32, count=1)
    ]

    pc_msg.fields = fields
    pc_msg.is_bigendian = False
    pc_msg.point_step = 12
    pc_msg.row_step = pc_msg.point_step * len(pc_data)
    pc_msg.is_dense = True
    pc_msg.data = pc_data.tobytes()
    pc_msg.height = 1
    pc_msg.width = len(pc_data)
    return pc_msg

def camera_orientation_matrix(pos, angle):
    """
    Creates matrix representaiton of SE(3) transformation about the y-axis.

    Params:
        pos (np.ndarray): The translation component
        angle (float): The angle to rotate by

    Returns:
        mat (np.ndarray): SE(3) matrix representation
    """
    mat = np.eye(4)
    mat[-1, :3] = pos
    mat[0, 0] = np.sin(angle)
    mat[0, 2] = np.cos(angle)
    mat[2, 0] = np.cos(angle)
    mat[2, 2] = -np.sin(angle)
    return mat

def make_frame(loc):
    """
    Helper function for creating coordinate frame meshes for visualization

    Params:
        loc (ndarray): position of coordinate frame

    Returns:
        TriangleMesh: mesh representing a depiction of coordinate axes at loc
    """
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    frame.translate(loc)
    return frame

class SceneGenerator:
    """
    An object that renders 3D pointclouds from segmentation masks and depth predictions.

    Params:
        checkpoint (str): The checkpoint for the DPT transformer depth predictor
        sensor_info_path (str): Path where the camera sensor data is stored
        depth_scale (float): Proportional depth prediction scaling
        depth_mode (str): Transformation of raw depth prediction (linear, square, cubic, exp)
        depth_thresh (int): Threshold used to extract objects from alpha channel of
                            foreground prediction
        depth_inf (float): Maximum depth prediction, used mainly for visualization
        downsample (float): Voxel size for voxel downsampling of pointclouds
                            (None for no downsampling)
        discrete_img (bool): Whether to use integer or floating point representation
                             of image for depth prediction
        flip_colors (bool): Whether to inverse the order of color channels
        flip_x (bool): Whether to flip the direction of the x axis (unused for now)
        icp_threshold (float): Convergence threshold parameter for ICP registration
        robust_kernel (bool): Whether to perform ICP with IWLS regression using a robust kernel
        tukey_sigma (float): Lengthscale parameter for the Tukey kernel for robust ICP
        num_clusters (int): Number of clusters to model in pointcloud clustering
        max_seg_objects (int): Maximum number of segmented pointclouds to consider for a given frame
        min_landmark_size (int): Threshold (in number of points) under which to ignore pointcloud
                                 clusters
        mesh_alpha (float): Smoothing parameter for mesh generation
        sfm (str or None): When not None, it is interpreted as the path to a checkpoint for a SfM
                           depth model. When specified, the corresponding SfM depth model will be
                           used for depth estimation rather than the vision transformer. When None,
                           vision transformer is used for depth estimation (default).
    """

    def __init__(
        self,
        checkpoint: str = 'vinvino02/glpn-nyu',
        sensor_info_path: str = 'sensor_info.p',
        depth_scale: float = 2.0,
        depth_mode: str = 'linear',
        depth_thresh: int = 50,
        depth_inf: float = 5.0,
        downsample: Optional[float] = 0.08,
        discrete_img: bool = False,
        flip_colors: bool = True,
        flip_x: bool = False,
        icp_threshold: float = 0.02,
        robust_kernel: bool = True,
        tukey_sigma: float = 0.1,
        num_clusters: int = 6,
        max_seg_objects: int = 4,
        min_landmark_size: int = 90,
        mesh_alpha: float = 0.25,
        sfm: Optional[str] = None,
    ):
        self.checkpoint = checkpoint
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        with open(sensor_info_path, 'rb') as f:
            self.sensor_info = pickle.load(f)

        self.intrinsics = ros_camera_intrinsics(self.sensor_info[SensorSource.FRONT_CAMERA])
        self.intrinsics_inv = np.linalg.inv(self.intrinsics)

        self.intrinsics_inv = torch.tensor(self.intrinsics_inv).to(self.device).float()

        if sfm is None:
            self.depth_model = (AutoModelForDepthEstimation.from_pretrained(checkpoint)
                                .to(self.device))
        else:
            self.depth_model = SfMDepthEstimator(sfm, device=self.device)

        self.img = None
        self.depth = None

        self.depth_scale = depth_scale
        self.depth_mode = depth_mode
        self.depth_thresh = depth_thresh
        self.depth_inf = depth_inf

        self.downsample = downsample

        self.discrete_img = discrete_img
        self.flip_colors = flip_colors
        self.flip_x = flip_x

        self.seg_session = new_session()

        self.icp_threshold = icp_threshold
        self.robust_kernel = robust_kernel
        self.tukey_sigma = tukey_sigma
        if self.robust_kernel:
            self.icp_model = lambda: p2p(
                o3d.pipelines.registration.TukeyLoss(k=self.tukey_sigma)
            )
        else:
            self.icp_model = lambda: p2p()

        self.num_clusters = num_clusters
        self.max_seg_objects = max_seg_objects
        self.min_landmark_size = min_landmark_size

        self.mesh_alpha = mesh_alpha

    def use_robust_kernel(self, opt: bool = True):
        self.robust_kernel = opt
        if self.robust_kernel:
            self.icp_model = lambda: p2p(
                o3d.pipelines.registration.TukeyLoss(k=self.tukey_sigma)
            )
        else:
            self.icp_model = lambda: p2p()

    def preprocess_img(self, img: np.ndarray) -> torch.Tensor:
        """
        Preprocess numpy image matrix for compatibility with depth prediction network.
        """
        while len(img.shape) < 2:
            img = np.expand_dims(img, 0)
        if img.shape == 2:
            img = np.stack([cv2.imdecode(im, cv2.IMREAD_UNCHANGED) for im in img])
        while len(img.shape) < 4:
            img = np.expand_dims(img, 0)
        if self.flip_colors:
            img = img[:, :, :, [2, 1, 0]]
        img_t = torch.Tensor(img).permute(0, 3, 1, 2)
        if self.discrete_img:
            return img_t
        return (img_t / 255.0).to(self.device)

    def predict_depth(self, img):
        """
        Predict depth map from image.

        Params:
            img (tensor): (Batch of) image data, dimensions [B, 3, W, H]

        Returns:
            depth (tensor): Real-valued depth map, dimensions [B, W, H]
        """
        with torch.no_grad():
            return self.depth_model(img).predicted_depth

    def predict_mask(self, img):
        """
        Predict background mask.

        Params:
            img (ndarray): Image pixels, dimensions [W, H, 3]

        Returns:
            mask (ndarray): Integer-valued alpha channel, lower values indicating background.
        """
        return remove(img, session=self.seg_session)[:, :, -1]

    def masked_depth_image(self, img):
        """
        Compute depth map, replacing background pixels with maximum depth.

        Params:
            img (ndarray): Image pixels, dimensions [W, H, 3]

        Returns:
            masked depth (ndarray)
        """
        img_t = self.preprocess_img(img)
        depth = self.predict_depth(img_t).squeeze().detach().cpu().numpy()
        mask = self.predict_mask(img)
        return np.where(mask < self.depth_thresh, self.depth_inf, depth)

    def preview(self, img, masked=True):
        """
        Displays a side-by-side view of the raw image and the predicted depth.

        Params:
            img (ndarray): The image to predict depth from
            masked (bool, True): Whether to eliminate background pixels from depth
        """
        if masked:
            depth_img = self.masked_depth_image(img)
        else:
            img_t = self.preprocess_img(img)
            depth_img = self.predict_depth(img_t).squeeze().detach().cpu().numpy()
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img)
        axs[0].set_title("Raw Image")
        axs[1].imshow(depth_img)
        axs[1].set_title("Predicted Depth")
        fig.show()

    def show_multi_image(self, imgs, mode='raw'):
        """
        Displays images (or depth predictions) from all cameras.

        Params:
            imgs (dict): Dictionary mapping camera key (see CAMERA_KEYS) to image pixels
            mode (str, 'raw'): Either 'raw', 'depth', or 'masked'. When 'raw', shows the
                               raw images from each camera. When 'depth', shows the depth
                               predictions for each camera. When 'masked', shows the depth
                               predictions with background pixels eliminated for each camera.
        """
        n = len(list(imgs.keys()))
        fig, axs = plt.subplots(1, n)
        for (i, key) in enumerate(imgs.keys()):
            img = imgs[key]
            if mode == 'depth':
                img = self.predict_depth(self.preprocess_img(img)).detach().cpu().numpy().squeeze()
            elif mode == 'masked':
                img = self.masked_depth_image(img)
            axs[i].imshow(img)
            axs[i].set_axis_off()
            axs[i].set_title(key)
        fig.tight_layout()
        fig.show()
        return fig

    @torch.no_grad()
    def pixel_coords_to_3d_coords(self, pixel_coords, depth_map=None, flat=False):
        """
        Transforms a list of 2D pixel coordinates to 3D coordinates using a depth map.

        When `depth_map` is `None`, the predicted depth map from `self.img` is used.

        When `flat` is `True`, the pixel coordinates are simply projected according to the
        camera matrix, and depth is ignored.
        """
        if depth_map is None:
            depth_map = self.depth
        depths = depth_map.squeeze()[pixel_coords[:, 0], pixel_coords[:, 1]]

        depths = torch.tensor(depths).float().to(self.device)

        pixel_coords = torch.tensor(pixel_coords).to(self.device).float()
        n = pixel_coords.shape[0]
        ones = torch.ones(n).unsqueeze(-1).to(self.device).float()
        homogeneous_pixel_coords = torch.cat([pixel_coords, ones], axis=-1)
        projected_pixel_coords = (self.intrinsics_inv @ homogeneous_pixel_coords.T).T

        # Correction for o3d
        projected_pixel_coords = projected_pixel_coords[:, [1, 0, 2]]
        projected_pixel_coords[:, 1] *= -1
        projected_pixel_coords[:, 0] *= -1

        if flat:
            return projected_pixel_coords
        return depths[:, None] * projected_pixel_coords

    def masked_pointcloud(self, img, flat=False, inv_depth=False) -> List[o3d.geometry.PointCloud]:
        """
        Returns a point cloud from obstacle pixels in `img`.
        When `self.downsample` is not `None`, the pointclouds will be downsampled
        via voxel downsampling to a resolution of `self.downsample`.

        When `flat` is `True`, the depth map is ignored.

        When `inv_depth` is `True`, depth values are replaced by `self.depth_inf - depth`
        (this was used for debugging).
        """
        mask = self.predict_mask(img)
        pixel_coords = np.argwhere(mask > self.depth_thresh)
        img_t = self.preprocess_img(img)
        depth = self.predict_depth(img_t)
        if self.depth_mode == 'square':
            depth = depth ** 2
        elif self.depth_mode == 'cubic':
            depth = depth ** 3
        elif self.depth_mode == 'exp':
            depth = torch.exp(depth)
        depth *= self.depth_scale
        if inv_depth:
            depth = self.depth_inf - depth
        points = (self.pixel_coords_to_3d_coords(pixel_coords, depth_map=depth, flat=flat)
                  .detach().cpu().numpy())
        pc = tensor_to_pointcloud(points)
        if self.downsample is None:
            return pc
        return pc.voxel_down_sample(voxel_size=self.downsample)

    def multi_masked_pointcloud(self, imgs, inv_depth=False, separate=False):
        """
        Returns a list of point clouds corresponding to observations from multiple cameras
        (see `self.masked_pointcloud`).

        Params:
            imgs (dict): Dictionary mapping camera keys (see CAMERA_KEYS) to image pixels.
            inv_depth (bool): Deprecated, kept for debugging (should be False)
            separate (bool): Deprecated, kept for debugging (should be False)

        Returns:
            pc (o3d.geometry.PointCloud): point cloud comprised of observations from all cameras,
            given with respect to the local frame.
        """
        pcs = []
        for camera_key in imgs:
            img_cur = imgs[camera_key]
            pc = self.masked_pointcloud(img_cur, inv_depth=inv_depth)
            camera_pose = CAMERA_LINKS[camera_key]
            theta = camera_pose['angle']
            xyz = camera_pose['pos_level']
            R = pc.get_rotation_matrix_from_xyz((0, theta, 0))
            pc = copy.deepcopy(pc).translate(xyz).rotate(R, center=xyz)
            pcs.append(pc)

        if not separate:
            pc = o3d.geometry.PointCloud()
            for p in pcs:
                pc.points.extend(p.points)
            return pc
        return pcs

    def find_landmarks(self, imgs, pc=None):
        """
        Identifies landmarks (i.e., large objects) in the current multi-camera observation.

        Params:
            imgs (dict): Dictionary of images mapping CAMERA_SOURCE to RGB image from which to
            find landmarks.
            pc (o3d.geometry.PointCloud, None): when not `None`, this pointcloud is searched
            for landmarks as opposed to the images in `imgs`.

        Returns:
            List[Landmark]: list of up to `self.max_seg_objects` landmarks, filtered by size.
        """
        if pc is None:
            pc = self.multi_masked_pointcloud(imgs, separate=False)
        clusterer = KMeans(self.num_clusters)
        all_points = np.array(pc.points)
        labels = clusterer.fit_predict(all_points)
        unique_labels = np.unique(labels)
        counts = [np.sum(labels == i) for i in unique_labels]
        # sort clusters in decreasing order of point count
        largest_labels = sorted(zip(unique_labels, counts), key=lambda x: x[-1])
        largest_labels = list(
            itertools.takewhile(lambda x: x[-1] >= self.min_landmark_size, largest_labels)
        )
        largest_labels = [label[0] for label in largest_labels[:self.max_seg_objects]]
        landmarks = []
        for label in largest_labels:
            pointcloud = all_points[labels == label]
            centroid = np.mean(pointcloud, axis=0)
            landmarks.append(Landmark(centroid, pointcloud))
        return landmarks

    def compute_scene_objects(
        self,
        imgs,
        include_landmarks=True,
        mesh=False,
        mesh_downsample=False,
        origin_frame=True,
    ):
        """
        Generates pointclouds and/or meshes of landmarks identified in a scene specified by images
        from the camera sensors.

        Params:
            imgs (dict): Mapping from CAMERA_KEYS to RGB images
            include_landmarks (bool): When True, predict landmarks and create meshes for
                                      coordinate axes representing their local frames.
            mesh (bool): When True, compute meshes of pointclouds.
            mesh_downsample (bool): When False, the pointclouds are not downsampled before
                                    computing the mesh.
            origin_frame (bool): When True, include mesh representation of axes at the agent's
                                 local frame.

        Returns:
            objs (List[Union[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud]]):
                Meshes and/or pointclouds of desired objects.
        """
        if origin_frame:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        else:
            frame = o3d.geometry.PointCloud()
            frame.points = o3d.utility.Vector3dVector(np.array([[0., 0., 0.]]))

        if mesh:
            downsample = self.downsample
            if not mesh_downsample:
                self.downsample = None
            pcs_sep = self.multi_masked_pointcloud(imgs, separate=True)
            meshes = [
                o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pc, self.mesh_alpha)
                for pc in pcs_sep
            ]
            self.downsample = downsample
            for mesh in meshes:
                mesh.compute_vertex_normals()
            pcs = o3d.geometry.PointCloud()
            for pc in pcs_sep:
                pcs.points.extend(pc.points)
            if not mesh_downsample:
                pcs = pcs.voxel_down_sample(voxel_size=self.downsample)
        else:
            pcs = self.multi_masked_pointcloud(imgs, separate=False)
            meshes = [pcs]

        if not include_landmarks:
            return [frame] + meshes

        landmarks = self.find_landmarks(imgs, pc=pcs)
        centroids = [landmark.centroid for landmark in landmarks]
        centroid_frames = [make_frame(centroid) for centroid in centroids]
        return [frame] + meshes + centroid_frames

    def scene(self, img, separate=False, inv_depth=False, include_landmarks=True, mesh=False):
        """
        Generates pointclouds from input images
        and launches a 3D visualization of these pointclouds.
        """
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        if isinstance(img, dict):
            pcs = self.multi_masked_pointcloud(img, separate=separate, inv_depth=inv_depth)
            if not separate:
                objects = self.compute_scene_objects(
                    img,
                    include_landmarks=include_landmarks,
                    origin_frame=not include_landmarks,
                    mesh=mesh
                )
                o3d.visualization.draw_geometries(objects, mesh_show_back_face=mesh)
            else:
                o3d.visualization.draw_geometries([frame] + pcs)
        else:
            pcs = self.masked_pointcloud(img, inv_depth=inv_depth)
            o3d.visualization.draw_geometries([frame, pcs])

    def scene_anim_old(self, imgs, inv_depth=False, n_frames=100):
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
        pcs = self.multi_masked_pointcloud(imgs, inv_depth=inv_depth)
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(axis)
        for pc in pcs:
            vis.add_geometry(pc)

        frames = []
        for frame_idx in range(n_frames):
            vis.clear_geometries()
            R = axis.get_rotation_matrix_from_xyz((0, 2 * np.pi * frame_idx / n_frames, 0))
            rotated_axis = copy.deepcopy(axis).rotate(R, center=(0, 0, 0))
            vis.add_geometry(rotated_axis)
            for pc in pcs:
                rotated_pc = copy.deepcopy(pc).rotate(R, center=(0, 0, 0))
                vis.add_geometry(rotated_pc)

            # vis.update_geometry()
            vis.poll_events()
            vis.update_renderer()

            camera = vis.get_view_control().convert_to_pinhole_camera_parameters()
            camera.extrinsic = np.eye(4) * self.depth_scale
            vis.get_view_control().convert_from_pinhole_camera_parameters(camera)
            breakpoint()
            frame_image = vis.capture_screen_float_buffer()
            frames.append((np.asarray(frame_image) * 255).astype(np.uint8))

        vis.destroy_window()
        gif_path = "~/scene.gif"
        imageio.mimsave(gif_path, frames, duration=1.0)
        return frames

    def scene_anim(
        self,
        imgs,
        n_frames=100,
        include_landmarks=True,
        mesh=False,
        path="~/scene.gif"
    ):
        """
        Renders a 3d scene from the observations from cameras specified by imgs and generates a
        gif animation. Animation is saved to `path`.

        Params:
            imgs (dict): Mapping from CAMERA_KEYS to RGB images.
            n_frames (int): Number of frames for animation.
            include_landmarks (bool): Whether to draw coordinate axes at the identified landmarks'
                                      local frames.
            mesh (bool): Whether to render meshes for reconstructed pointclouds.
            path (str): Filepath to save gif animation.

        Returns:
            frames (List[np.ndarray]): The rendered frames.
        """
        objects = self.compute_scene_objects(
            imgs,
            include_landmarks=include_landmarks,
            origin_frame=not include_landmarks,
            mesh=mesh
        )
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for obj in objects:
            vis.add_geometry(obj)

        frames = []
        for frame_idx in range(n_frames):
            vis.clear_geometries()
            R = objects[0].get_rotation_matrix_from_xyz((0, 2 * np.pi * frame_idx / n_frames, 0))
            for obj in objects:
                rotated_obj = copy.deepcopy(obj).rotate(R, center=(0, 0, 0))
                vis.add_geometry(rotated_obj)

            vis.poll_events()
            vis.update_renderer()

            camera = vis.get_view_control().convert_to_pinhole_camera_parameters()
            camera.extrinsic = np.eye(4) / self.depth_scale
            vis.get_view_control().convert_from_pinhole_camera_parameters(camera)
            frame_image = vis.capture_screen_float_buffer()
            frames.append((np.asarray(frame_image) * 255).astype(np.uint8))

        vis.destroy_window()
        gif_path = path
        imageio.mimsave(gif_path, frames, duration=1.0)
        return frames

    def odometry(self, video_data, start_frame=0, max_frames=10, frame_stride=1):
        """
        Computes odometry estimates from video sequence from all cameras.

        Params:
            video_data (dict): Mapping from CAMERA_KEYS to sequence of RGB images
            start_frame (int): Time index to start odometry estimate from
            max_frames (int): Time horizon to estimate odometry over
            frame_stride (int): Number of frames between subsequent images for odometry estimates

        Returns:
            transforms (List[np.ndarray]): List of SE(3) matrix representations of pose
                                           transformations over the specified horizon.
        """
        ti = start_frame
        tf = min(start_frame + max_frames + 1, len(video_data['FRONT_CAMERA']))
        obs_data_n = {
            key: np.array(video_data[key])[ti:tf:frame_stride].reshape(-1, 800, 800, 3)
            for key in CAMERA_KEYS
        }
        obs_data = {
            key: self.preprocess_img(obs_data_n[key]) for key in obs_data_n
        }

        # [t, cam, C, W, H]
        # obs_data_n = np.stack([obs_data_n[k] for k in obs_data], axis=1)
        obs_data = torch.stack([obs_data[k] for k in obs_data], axis=1)

        H = obs_data.shape[0]

        transforms = []

        pc_tm1 = self.multi_masked_pointcloud({k: obs_data_n[k][0] for k in CAMERA_KEYS}, flat=True)
        for t in tqdm(range(1, H)):
            pc_t = self.multi_masked_pointcloud(
                {k: obs_data_n[k][t] for k in CAMERA_KEYS},
                flat=True
            )
            pc_t.estimate_normals()
            transforms.append(o3d.pipelines.registration.registration_icp(
                pc_tm1, pc_t, self.icp_threshold, np.eye(4),
                self.icp_model()
            ).transformation)
            pc_tm1 = pc_t

        return transforms

    def integrate_transforms(self, transforms, x0=np.eye(4)[-1], alpha=1.0):
        """
        Applies a sequence of SE(3) transformations (as returned by `self.odometry`) to a
        starting state.

        Params:
            transforms (List[np.ndarray]): List of SE(3) matrix representations
            x0 (np.ndarray): Initial pose
            alpha (float): Exponential smoothing parameter for transformations

        Returns:
            x (List[np.ndarray]): The sequence of planar coordinates due to the sequence of
                                  transforms.
        """
        states = [x0]
        trans = transforms[0]
        for T in transforms:
            trans = (1 - alpha) * trans + alpha * T
            states.append(trans @ states[-1])
        x = np.stack(states)
        x[:, 2] *= -1
        return x[:, [1, 2]], x

    def preview_odometry(self, video_data, start_frame=0, max_frames=10, frame_stride=1):
        """
        Estimates odometry from video_data and plots the resulting planar trajectory.

        Params:
            video_data (dict): Mapping from CAMERA_KEYS to sequence of RGB images
            start_frame (int): The time index to start odometry estimation from
            max_frames (int): The time horizon to estimate odometry over
            frame_stride (int): The number of frames between subsequent images for odometry
                                estimation.

        Returns:
            dict: Dictionary containing computed planar coordinates at key 'positions', ground truth
                  odometry data at key 'gt', and inferred pose transformations at key 'transforms'.
        """
        ti = start_frame
        tf = min(start_frame + max_frames + 1, len(video_data['position']))

        gt_odo = np.array(video_data['position'])[ti:tf]

        fig, axs = plt.subplots(1, 2)
        self.plot_trajectory(axs[0], gt_odo)
        axs[0].set_title("Ground truth positions")

        transforms = self.odometry(
            video_data,
            start_frame=start_frame,
            max_frames=max_frames,
            frame_stride=frame_stride
        )

        positions, pos_all = self.integrate_transforms(transforms)

        self.plot_trajectory(axs[1], positions)
        axs[1].set_title("Predicted positions")
        fig.tight_layout()
        fig.show()

        return {
            'positions': positions,
            'gt': gt_odo,
            'positions_all': pos_all,
            'transforms': transforms,
        }

    def plot_trajectory(self, ax, traj):
        ax.plot(traj[:, 0], traj[:, 1])
        ax.scatter([traj[0, 0]], [traj[0, 1]], marker='o', label='Start')
        ax.scatter([traj[-1, 0]], [traj[-1, 1]], marker='x', label='End')
        ax.legend()

    @torch.no_grad()
    def analyze_depth(
        self,
        data_path='data_dump_still.p',
        maxlen=100,
        start_frame=0,
        batch_size=16,
        seed=42,
        num_pixels=100,
        bootstrap_samples=100,
        aggregate=False,
        compare_with_sfm=False,
    ):
        """
        Performs robustness analysis of depth estimator.
        This involves two experiments:

            1. For a bunch of randomly chosen pixels, estimate the depth prediction at those
               pixels over some time horizon where the camera is held fixed. Plot the evolution
               of depth estimates for the pixel with the largest variance over the horizon.
            2. Use bootstrapping to estimate the distribution over depth standard deviations over
               the horizon.

        Params:
            data_path (str): Path to dataset pickle containing a stream of images from each camera
                             from a fixed pose.
            maxlen (int): Length of time horizon to compute evolution of depth predictions.
            start_frame (int): Time index to start computing depth estimates.
            batch_size (int): Batch size for batched depth inference.
            seed (int): Random seed for random pixel selection and bootstrapping.
            num_pixels (int): Number of random pixels (without replacement) to examine.
            bootstrap_samples (int): Number of bootstrap standard deviation estimates to draw.
            aggregate (bool): Whether to include results aggregated over all cameras.

        Returns:
            dict: Dictionary with generated figures. The 'fig' key contains the figure for the
                  analysis computed over each camera separately, while the 'aggregated_fig' contains
                  the figure for the analysis of the data aggregated from all cameras.
        """
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        if compare_with_sfm:
            sfm_data = np.load('sfm_depth_data.npz')['sfm_depths']

        rng = np.random.default_rng(seed=seed)
        pixel_coords = rng.choice(800 * 300, size=num_pixels, replace=False)
        pixel_coords = np.stack([[c // 800, c % 800] for c in pixel_coords])

        end_frame = min(start_frame + maxlen, len(data[CAMERA_KEYS[0]]))
        ts = np.arange(start_frame, end_frame) / 30

        obs_data = {
            key: list(data[key])[start_frame:end_frame] for key in CAMERA_KEYS
        }

        N = len(obs_data[CAMERA_KEYS[0]])
        num_batches = int(math.ceil(N / batch_size))

        pixel_depths = {
            key: [] for key in obs_data
        }

        for i in tqdm(range(num_batches)):
            start = i * batch_size
            end = min((i + 1) * batch_size, N)
            batches = {
                key: np.stack([
                    cv2.imdecode(x, cv2.IMREAD_UNCHANGED)
                    for x in obs_data[key][start:end]
                ])
                for key in CAMERA_KEYS
            }
            batches = {
                key: self.preprocess_img(batches[key])
                for key in batches
            }
            depths = {
                key: self.predict_depth(batches[key]) for key in batches
            }
            _pixel_depths = {
                key: depths[key][:, pixel_coords[:, 0], pixel_coords[:, 1]].detach().cpu().numpy()
                for key in depths
            }
            for key in pixel_depths:
                pixel_depths[key].extend(_pixel_depths[key])

        def plot_best_worst_depths(depths: np.ndarray, ax, camera_key=None, debug=False):
            pd_stds = np.std(depths, axis=0)
            best_idx = np.argmin(pd_stds)
            worst_idx = np.argmax(pd_stds)
            if debug:
                breakpoint()

            if compare_with_sfm:
                sfm_stds = np.std(sfm_data, axis=1)
                sfm_worst_idx = np.argmax(sfm_stds)
                ax.plot(ts, depths[:, worst_idx], linewidth=4, label='VT', color='blue')
                ax.plot(ts, sfm_data[sfm_worst_idx, :], linewidth=4, label='SfM', color='red')
            else:
                ax.plot(
                    ts,
                    depths[:, best_idx],
                    linewidth=4,
                    label='Pixel with least depth variance'
                )
                # ax.plot(ts, depths[:, worst_idx], linewidth=4, label='Pixel with largest depth variance')
            ax.legend()
            if camera_key is None:
                ax.set_ylabel("Depth (Meters)")
            else:
                ax.set_ylabel(f"{camera_key}\nDepth (Meters)")

        def plot_bootstrap_depths(depths: np.ndarray, ax, debug=False):
            bootstrap_vals = []
            N = depths.shape[0]
            pd_stds = np.std(depths, axis=0)
            min_std = np.min(pd_stds)
            max_std = np.max(pd_stds)
            for _ in range(bootstrap_samples):
                bootstrap_indices = rng.choice(N, size=N, replace=True)
                bootstrap_stds = pd_stds[bootstrap_indices]
                bootstrap_vals.append(np.mean(bootstrap_stds))

            if debug:
                breakpoint()
            ax.hist(bootstrap_vals, density=True, label='VT', color='blue', alpha=0.5)
            if compare_with_sfm:
                sfm_stds = np.std(sfm_data, axis=1)
                N_sfm = sfm_data.shape[0]
                sfm_bootstrap_vals = []
                for _ in range(bootstrap_samples):
                    bootstrap_indices = rng.choice(N, size=N, replace=True)
                    bootstrap_stds = sfm_stds[bootstrap_indices]
                    sfm_bootstrap_vals.append(np.mean(bootstrap_stds))
                ax.hist(sfm_stds, density=True, label='SfM', color='red', alpha=0.5)
                # ax.hist(sfm_bootstrap_vals, density=True, label='SfM', color='red', alpha=0.5)
                ax.legend()

            ax.set_ylabel('Probability Density')
            # ax.set_xlim(min_std, max_std)
            # ax.set_xlim(0.03, 0.06)
            ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 100))
            ax.xaxis.set_major_formatter(ticks)

        fig, axs = plt.subplots(len(CAMERA_KEYS), 2)
        for (i, key) in enumerate(pixel_depths.keys()):
            plot_best_worst_depths(np.array(pixel_depths[key]), axs[i, 0], camera_key=key)
            plot_bootstrap_depths(np.array(pixel_depths[key]), axs[i, 1])

        axs[-1, 0].set_xlabel('Time (s)')
        axs[0, 0].set_title("Fluctation of Depth Estimates")
        axs[-1, 1].set_xlabel("Depth Standard Deviation (cm)")
        axs[0, 1].set_title("Bootstrapped Pixel Depth Standard Deviation")
        fig.tight_layout()

        aggregated_fig, aggregated_axs = plt.subplots(1, 2)
        aggregated_pds = np.concatenate(
            [np.array(pixel_depths[key]) for key in pixel_depths],
            axis=1
        )
        plot_best_worst_depths(aggregated_pds, aggregated_axs[0], debug=False)
        plot_bootstrap_depths(aggregated_pds, aggregated_axs[1], debug=False)
        aggregated_axs[0].set_xlabel('Time (s)')
        aggregated_axs[0].set_title('Fluctuation of Depth Estimates')
        aggregated_axs[0].margins(0)
        if not compare_with_sfm:
            aggregated_axs[0].set_ylim(1.5, 5.0)
        aggregated_axs[1].set_xlabel('Depth Standard Deviation (cm)')
        aggregated_axs[1].set_title('Bootstrapped Pixel Depth Standard Deviation')
        aggregated_fig.tight_layout()

        return {
            'pixel_depths': pixel_depths,
            'fig': fig,
            'aggregated_fig': aggregated_fig,
            'aggregated_axs': aggregated_axs,
            'axs': axs
        }

def main(args):
    kwargs = dict()
    if args.sfm:
        kwargs['sfm'] = args.sfm_ckpt

    sg = SceneGenerator(
        args.ckpt,
        sensor_info_path=args.sensor_info_path,
        depth_scale=args.depth_scale,
        **kwargs
    )

    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    img_multi = {key: data[key][0] for key in CAMERA_KEYS}

    img_test = data['FRONT_CAMERA'][2]

    with open(args.video_data_path, 'rb') as f:
        video_data = pickle.load(f)

    data_dict = {
        'img_multi': img_multi,
        'img_test': img_test,
        'video_data': video_data,
        'data': data
    }

    job_lookup = {
        'analyze-depth': depth_job,
        'scene': scene_job,
        'stream-scene': stream_scene_job,
        'scene-anim': scene_anim_job,
    }

    job = dummy_job if args.job not in job_lookup else job_lookup[args.job]

    res = job(sg, args, data_dict)

    breakpoint()
    return

def depth_job(sg, args, data):
    """
    Conducts depth estimate robustness experiment. See `SceneGenerator.analyze_depth`.
    """
    return sg.analyze_depth(
        maxlen=args.depth_maxlen,
        batch_size=16,
        num_pixels=args.depth_num_pixels,
        bootstrap_samples=args.depth_bootstrap_samples,
        start_frame=args.depth_start_frame,
    )

def scene_job(sg, args, data):
    """
    Renders reconstructed 3D scene. See `SceneGenerator.scene`.
    """
    return sg.scene(
        data['img_multi'],
        separate=False,
        mesh=args.scene_mesh,
        include_landmarks=not args.no_landmarks
    )

def stream_scene_job(sg, args, data):
    """
    Computes pointclouds and landmarks from camera images and publishes them
    to the `/landmark_pointclouds` ROS topic.
    """
    import rospy
    rospy.init_node('stream-scene-job')
    pub = rospy.Publisher('/landmark_pointclouds', PointCloud2, queue_size=10)
    pc = sg.multi_masked_pointcloud(data['img_multi'], separate=False)
    rospc = o3d_to_ros_pointcloud(pc)
    rate = rospy.Rate(10)
    rospy.loginfo("Streaming pointcloud...")
    while True:
        rospc.header.stamp = rospy.Time.now()
        pub.publish(rospc)
        rate.sleep()

def scene_anim_job(sg, args, data):
    """
    Renders an animation depicting the reconstructed 3D scene.
    See `SceneGenerator.scene_anim`.
    """
    return sg.scene_anim(
        data['img_multi'],
        n_frames=args.scene_frames,
        mesh=args.scene_mesh,
        include_landmarks=not args.no_landmarks
    )

def dummy_job(*args):
    """
    Dummy to specify no action (return straight to Python shell).
    """
    return None

class SfMDepthEstimator:
    """
    Wrapper class for SfM depth estimation.
    """

    def __init__(self, ckpt, device='cuda'):
        """
        Params:
            ckpt (str): Location of torch checkpoint for SfM disp_net.
            device (str): Torch device to place model.
        """
        self.ckpt = ckpt
        self.device = device

        disp_net = DispNetS().to(self.device)
        weights = torch.load(self.ckpt)
        disp_net.load_state_dict(weights['state_dict'])
        disp_net.eval()
        self.model = disp_net

    @torch.no_grad()
    def __call__(self, img):
        out = self.model(img).squeeze()
        return DepthEstimateWrapper(1 / out)

@dataclasses.dataclass
class DepthEstimateWrapper:
    """
    Wrapper class to unify API between vision transformer and SfM.
    """
    predicted_depth: np.ndarray


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--ckpt-path", type=str, default='checkpoints/13000')
    parser.add_argument("--ckpt", type=str, default='vinvino02/glpn-nyu')
    parser.add_argument("--data-path", type=str, default='sample-data-scattered.p')
    parser.add_argument("--video-data-path", type=str, default='data_dump.p')
    parser.add_argument('--sensor-info-path', type=str, default='sensor_info.p')
    parser.add_argument("--dryrun", action="store_true", default=False, help="Do not automatically load SAM and segment images")
    parser.add_argument("--job", default=None)
    parser.add_argument("--depth-scale", type=float, default=2.0)
    parser.add_argument("--sfm", action='store_true', default=False)
    parser.add_argument("--sfm-ckpt", type=str, default='/opt/noetic_ws/checkpoints/14000/disp.pth.tar')
    depth_group = parser.add_argument_group("depth job")
    depth_group.add_argument("--depth-maxlen", type=int, default=100)
    depth_group.add_argument("--depth-num-pixels", type=int, default=512)
    depth_group.add_argument("--depth-bootstrap-samples", type=int, default=1000)
    depth_group.add_argument("--depth-start-frame", type=int, default=0)
    scene_group = parser.add_argument_group("scene generation")
    scene_group.add_argument("--scene-frames", type=int, default=100)
    scene_group.add_argument("--scene-mesh", action='store_true', default=False)
    scene_group.add_argument("--no-landmarks", action='store_true', default=False)

    args = parser.parse_args()
    main(args)
