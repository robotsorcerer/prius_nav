#!/usr/bin/env python3

import argparse
import copy
import cv2
import dataclasses
import imageio
import math
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pickle
import torch

from rembg import remove, new_session
from sklearn.cluster import KMeans
from transformers import AutoModelForDepthEstimation
from typing import List, Optional

from tqdm import tqdm

from scene_parser import SensorSource, SensorCollection, ros_camera_intrinsics

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
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def camera_orientation_matrix(pos, angle, axis):
    mat = np.eye(4)
    mat[-1, :3] = pos
    mat[0, 0] = np.sin(angle)
    mat[0, 2] = np.cos(angle)
    mat[2, 0] = np.cos(angle)
    mat[2, 2] = -np.sin(angle)
    return mat

def make_frame(loc):
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
        depth_thresh (int): Threshold used to extract objects from alpha channel of foreground prediction
        depth_inf (float): Maximum depth prediction, used mainly for visualization
        discrete_img (bool): Whether to use integer or floating point representation of image for depth prediction
        flip_colors (bool): Whether to inverse the order of color channels
        flip_x (bool): Whether to flip the direction of the x axis (unused for now)
    """
    def __init__(
        self,
        checkpoint: str = 'vinvino02/glpn-nyu',
        sensor_info_path: str = 'sensor_info.p',
        depth_scale: float = 1.0,
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
    ):
        self.checkpoint = checkpoint
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        with open(sensor_info_path, 'rb') as f:
            self.sensor_info = pickle.load(f)

        self.intrinsics = ros_camera_intrinsics(self.sensor_info[SensorSource.FRONT_CAMERA])
        self.intrinsics_inv = np.linalg.inv(self.intrinsics)

        self.intrinsics_inv = torch.tensor(self.intrinsics_inv).to(self.device).float()

        self.depth_model = AutoModelForDepthEstimation.from_pretrained(checkpoint).to(self.device)

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
            depth = depth **3
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

        Returns:
            pcs (List[o3d.geometry.PointCloud]): point cloud for each camera view, rotated
            and translated relative to the camera pose with respect to the car's local frame.
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
        if pc is None:
            pc = self.multi_masked_pointcloud(imgs, separate=False)
        clusterer = KMeans(self.num_clusters)
        all_points = np.array(pc.points)
        labels = clusterer.fit_predict(all_points)
        unique_labels = np.unique(labels)
        counts = [np.sum(labels == i) for i in unique_labels]
        # sort clusters in decreasing order of point count
        largest_labels = sorted(zip(unique_labels, counts), key=lambda x: -x[-1])
        largest_labels = [l[0] for l in largest_labels[:self.max_seg_objects]]
        landmarks = []
        for label in largest_labels:
            pointcloud = all_points[labels == label]
            centroid = np.mean(pointcloud, axis=0)
            landmarks.append(Landmark(centroid, pointcloud))
        return landmarks

    def scene(self, img, separate=False, inv_depth=False):
        """
        Generates pointclouds from input images
        and launches a 3D visualization of these pointclouds.
        """
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        if isinstance(img, dict):
            pcs = self.multi_masked_pointcloud(img, separate=separate, inv_depth=inv_depth)
            if not separate:
                landmarks = self.find_landmarks(img, pc=pcs)
                centroids = [landmark.centroid for landmark in landmarks]
                centroid_frames = [make_frame(centroid) for centroid in centroids]
                o3d.visualization.draw_geometries([frame, pcs] + centroid_frames)
            else:
                o3d.visualization.draw_geometries([frame] + pcs)
        else:
            pcs = self.masked_pointcloud(img, flat=flat, inv_depth=inv_depth)
            o3d.visualization.draw_geometries([frame, pcs])

    def scene_anim(self, imgs, inv_depth=False, n_frames=100):
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
            camera.extrinsic = np.eye(4)
            vis.get_view_control().convert_from_pinhole_camera_parameters(camera)
            breakpoint()
            frame_image = vis.capture_screen_float_buffer()
            frames.append((np.asarray(frame_image) * 255).astype(np.uint8))

        vis.destroy_window()
        gif_path = "~/scene.gif"
        imageio.mimsave(gif_path, frames, duration=1.0)
        return frames

    def odometry(self, video_data, start_frame=0, max_frames=10, frame_stride=1):
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
            pc_t = self.multi_masked_pointcloud({k: obs_data_n[k][t] for k in CAMERA_KEYS}, flat=True)
            pc_t.estimate_normals()
            transforms.append(o3d.pipelines.registration.registration_icp(
                pc_tm1, pc_t, self.icp_threshold, np.eye(4),
                self.icp_model()
            ).transformation)
            pc_tm1 = pc_t

        return transforms

    def integrate_transforms(self, transforms, x0=np.eye(4)[-1], alpha=1.0):
        states = [x0]
        trans = transforms[0]
        for T in transforms:
            trans = (1 - alpha) * trans + alpha * T
            states.append(trans @ states[-1])
        x = np.stack(states)
        x[:, 2] *= -1
        return x[:, [1, 2]], x

    def preview_odometry(self, video_data, start_frame=0, max_frames=10, frame_stride=1):
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
    ):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        rng = np.random.default_rng(seed=seed)
        pixel_coords = rng.choice(800 * 300, size=num_pixels, replace=False)
        pixel_coords = np.stack([[c // 800, c % 800] for c in pixel_coords])

        end_frame = min(start_frame + maxlen, len(data[CAMERA_KEYS[0]]))

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

        fig, axs = plt.subplots(len(CAMERA_KEYS), 2)
        for (i, key) in enumerate(pixel_depths.keys()):
            # Plot best and worst case fluctuations
            pds = np.array(pixel_depths[key])
            pd_stds = np.std(pds, axis=0)
            best_idx = np.argmin(pd_stds)
            worst_idx = np.argmax(pd_stds)
            axs[i, 0].plot(pds[:, best_idx], label='Pixel with least depth variance')
            axs[i, 0].plot(pds[:, worst_idx], label='Pixel with largest depth variance')
            axs[i, 0].legend()
            axs[i, 0].set_ylabel(f"{key}\nEstimated Depth")

            # Bootstrap estimate of fluctuation
            bootstrap_vals = []
            for _ in range(bootstrap_samples):
                bootstrap_indices = rng.choice(num_pixels, size=num_pixels, replace=True)
                bootstrap_stds = pd_stds[bootstrap_indices]
                bootstrap_vals.append(np.mean(bootstrap_stds))

            axs[i, 1].hist(bootstrap_vals, density=True)
            axs[i, 1].set_ylabel("Probability Density")

        axs[-1, 0].set_xlabel('Time')
        axs[0, 0].set_title("Fluctation of Depth Estimates")
        axs[-1, 1].set_xlabel("Depth Standard Deviation")
        axs[0, 1].set_title("Mean Pixel Depth Standard Deviation")
        fig.tight_layout()
        return {
            'pixel_depths': pixel_depths,
            'fig': fig,
            'axs': axs
        }

def main(args):
    sg = SceneGenerator(args.ckpt, sensor_info_path=args.sensor_info_path)
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    img_multi = {key: data[key][0] for key in CAMERA_KEYS}

    img_test = data['FRONT_CAMERA'][2]

    with open(args.video_data_path, 'rb') as f:
        video_data = pickle.load(f)

    data_dict = {'img_multi': img_multi, 'img_test': img_test, 'video_data': video_data, 'data': data}

    job = {
        'analyze-depth': depth_job,
        'scene': scene_job,
        'None': lambda _: None,
    }[args.job]

    res = job(sg, args, data_dict)

    breakpoint()
    return

def depth_job(sg, args, data):
    return sg.analyze_depth(
        maxlen=args.depth_maxlen,
        batch_size=16,
        num_pixels=args.depth_num_pixels,
        bootstrap_samples=args.depth_bootstrap_samples,
        start_frame=args.depth_start_frame
    )

def scene_job(sg, args, data):
    return sg.scene(data['img_multi'], separate=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--ckpt-path", type=str, default='checkpoints/13000')
    parser.add_argument("--ckpt", type=str, default='vinvino02/glpn-nyu')
    parser.add_argument("--data-path", type=str, default='sample-data-scattered.p')
    parser.add_argument("--video-data-path", type=str, default='data_dump.p')
    parser.add_argument('--sensor-info-path', type=str, default='sensor_info.p')
    parser.add_argument("--dryrun", action="store_true", default=False, help="Do not automatically load SAM and segment images")
    parser.add_argument("--job", default=None)
    depth_group = parser.add_argument_group("depth job")
    depth_group.add_argument("--depth-maxlen", type=int, default=100)
    depth_group.add_argument("--depth-num-pixels", type=int, default=512)
    depth_group.add_argument("--depth-bootstrap-samples", type=int, default=1000)
    depth_group.add_argument("--depth-start-frame", type=int, default=0)

    args = parser.parse_args()
    main(args)
