#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pickle
import torch

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from typing import List

from sfm.sfm_model import SfmLearner

from scene_parser import SensorSource, SensorCollection, ros_camera_intrinsics

def tensor_to_pointcloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

class SceneGenerator:
    """
    An object that renders 3D pointclouds from segmentation masks and depth predictions.
    """
    def __init__(
        self,
        checkpoint_path: str,
        sam_path: str = './models/sam_vit_h_4b8939.pth',
        sensor_info_path: str = 'sensor_info.p',
    ):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        with open(sensor_info_path, 'rb') as f:
            self.sensor_info = pickle.load(f)

        self.intrinsics = ros_camera_intrinsics(self.sensor_info[SensorSource.FRONT_CAMERA])
        self.intrinsics_inv = np.linalg.inv(self.intrinsics)

        self.intrinsics_inv = torch.tensor(self.intrinsics_inv).to(self.device).float()

        self.sfm = SfmLearner(self.intrinsics, checkpoint_path=checkpoint_path)
        self.sfm.disp_net.eval()
        self.sfm.pose_exp_net.eval()

        self.mask_indices = set()

        self.sam = sam_model_registry['vit_h'](checkpoint=sam_path)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)

        self.masks = None
        self.img = None
        self.depth = None

    def add_mask_index(self, index):
        """
        Add a mask index to be represented in the pointcloud.
        """
        self.mask_indices.add(index)

    def remove_mask_index(self, index):
        """
        Remove a mask index from the set of objects represented by the pointcloud.
        """
        self.mask_indices.remove(index)

    def set_mask_indices(self, i):
        """
        Sets the set of mask indices that will be generated in the pointcloud.
        """
        if isinstance(i, int):
            i = [i]
        self.mask_indices = set(i)

    def set_image(self, img):
        """
        Sets the image that the scene will be rendered from.
        """
        self.img = img
        self.masks = self.mask_generator.generate(img)
        self.depth = self.predict_depth(img)

    def predict_depth(self, img):
        """
        Uses the SfM model to predict the depthmap for `img`.
        """
        img = img / 255
        disp = (self.sfm.disp_net(torch.FloatTensor(img).unsqueeze(0).permute(0, 3, 1, 2))
                .squeeze()
                .detach()
                .cpu()
                .numpy())
        return 1 / disp

    def masked_image(self, mask_color=[255, 255, 0], mask_alpha=0.5):
        """
        Returns an image containing the pixels from `self.img` with pixels
        highlighted according to the pixels contained in the masks specified
        by the enabled mask indices. These are the indices specified by
        `self.set_mask_indices` or `self.add_mask_index`.
        """
        mask_color = np.array(mask_color)
        img_masked = self.img.copy()
        for i in self.mask_indices:
            mask = self.masks[i]['segmentation']
            img_masked[mask] = mask_alpha * mask_color + (1 - mask_alpha) * self.img[mask]
        return img_masked

    def preview(self, mask_color=[255, 255, 0], mask_alpha=0.5):
        """
        Displays a visualization of the pixels masked by `self.mask_indices` and
        the predicted depth map.
        """
        img_masked = self.masked_image(mask_color=mask_color, mask_alpha=mask_alpha)
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img_masked)
        axs[0].set_title("Masked Pixels")
        axs[1].imshow(self.depth)
        axs[1].set_title("Predicted Depth")
        fig.show()

    def pixel_coords_to_3d_coords(self, pixel_coords, depth_map=None, flat=False):
        """
        Transforms a list of 2D pixel coordinates to 3D coordinates using a depth map.

        When `depth_map` is `None`, the predicted depth map from `self.img` is used.

        When `flat` is `True`, the pixel coordinates are simply projected according to the
        camera matrix, and depth is ignored.
        """
        if depth_map is None:
            depth_map = self.depth
        depths = depth_map[pixel_coords[:, 0], pixel_coords[:, 1]]

        depths = torch.tensor(depths).float().to(self.device)

        pixel_coords = torch.tensor(pixel_coords).to(self.device).float()
        n = pixel_coords.shape[0]
        ones = torch.ones(n).unsqueeze(-1).to(self.device).float()
        homogeneous_pixel_coords = torch.cat([pixel_coords, ones], axis=-1)
        projected_pixel_coords = (self.intrinsics_inv @ homogeneous_pixel_coords.T).T

        # Correction for o3d
        projected_pixel_coords = projected_pixel_coords[:, [1, 0, 2]]
        projected_pixel_coords[:, 1] *= -1

        if flat:
            return projected_pixel_coords
        return depths[:, None] * projected_pixel_coords

    def masked_pointcloud(self, flat=False) -> List[o3d.geometry.PointCloud]:
        """
        Returns a list of 3D pointclouds for each segmentation mask in `self.mask_indices`.

        When `flat` is `True`, the depth map is ignored.
        """
        pcs = []
        for i in self.mask_indices:
            mask = self.masks[i]['segmentation']
            pixel_coords = np.argwhere(mask)
            points = (self.pixel_coords_to_3d_coords(pixel_coords, flat=flat)
                      .detach().cpu().numpy())
            pcs.append(tensor_to_pointcloud(points))
        return pcs

    def scene(self):
        """
        Generates pointclouds from the objects specified in `self.mask_indices`
        and launches a 3D visualization of these pointclouds.
        """
        pcs = self.masked_pointcloud()
        o3d.visualization.draw_geometries(pcs)

def main(args):
    sg = SceneGenerator(args.ckpt_path, sensor_info_path=args.sensor_info_path)
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)


    img_test = data['FRONT_CAMERA'][2]
    depth = sg.predict_depth(img_test)

    if not args.dryrun:
        print("Loading models, segmenting image, predicting depth...")
        sg.set_image(img_test)
        sg.add_mask_index(0)
        print("Done")

    breakpoint()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--ckpt-path", type=str, default='checkpoints/13000')
    parser.add_argument("--ckpt-path", type=str, default='./checkpoints/sfm')
    parser.add_argument("--data-path", type=str, default='sample-data-scattered.p')
    parser.add_argument('--sensor-info-path', type=str, default='sensor_info.p')
    parser.add_argument("--dryrun", action="store_true", default=False, help="Do not automatically load SAM and segment images")

    args = parser.parse_args()
    main(args)
