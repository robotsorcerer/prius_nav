import argparse
import cv2
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from functools import partial
from typing import List

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

def show_mask(mask, img, color=[255, 255, 0], alpha=1.0):
    img_masked = img.copy()
    img_masked[mask] = (1 - alpha) * img_masked[mask] + alpha * np.array(color)
    plt.imshow(img_masked)
    plt.show()

def get_depths_from_coords(coords, depth_map):
    return depth_map[coords[:, 0], coords[:, 1]]

def get_depths_from_mask(mask, depth_map):
    coords = jnp.argwhere(mask)
    return get_depths_from_coords(coords, depth_map), coords

@partial(jax.vmap, in_axes=(0, 0, None))
def pixel_to_R3(pixel_coord: jnp.ndarray, depth: float, intrinsics_inv: jnp.ndarray) -> jnp.ndarray:
    pixel_coord = jnp.concatenate([pixel_coord, jnp.array([1.0])])
    coord_flat = intrinsics_inv @ pixel_coord
    return depth * coord_flat

def get_pointcloud_from_mask(mask, depth_map, intrinsics_inv) -> o3d.geometry.PointCloud:
    depths, coords = get_depths_from_mask(mask, depth_map)
    r3_coords = pixel_to_R3(coords, depths, intrinsics_inv)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(r3_coords)
    return pcd

def render_pointclouds(pointclouds: List[o3d.geometry.PointCloud]):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for pc in pointclouds:
        vis.add_geometry(pc)
    vis.run()

def main(args):
    raw_image = cv2.imread('data/sample-image-01.png')
    depth_map = cv2.imread('data/sample-depth-01.png')
    depth_mask = jnp.array(depth_map).mean(axis=-1)
    sam = sam_model_registry['vit_h'](checkpoint='models/sam_vit_h_4b8939.pth')
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(raw_image)

    intrinsics = np.array([
        [476.703, 0.0, 400.5],
        [0.0, 476.703, 400.5],
        [0.0, 0.0, 1.0]
    ])

    intrinsics_inv = jnp.array(np.linalg.inv(intrinsics))

    # mask = jnp.array(masks[0]['segmentation'])
    # depths, coords = get_depths_from_mask(mask, depth_mask)
    # breakpoint()
    #
    # return

    pcds = [
        get_pointcloud_from_mask(jnp.array(mask['segmentation']), depth_mask, intrinsics_inv)
        for mask in masks[:args.max_masks]
    ]

    breakpoint()
    vis = o3d.visualization.draw_geometries(pcds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--iou-thresh', type=float, default=None)
    parser.add_argument('--max-masks', type=int, default=12)
    args = parser.parse_args()
    main(args)
