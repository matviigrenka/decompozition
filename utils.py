import os
from typing import Optional, Tuple

import numpy as np
import torch

from dataset import (
    estimate_normals,
    farthest_point_sampling,
    load_obj_mesh,
    load_off_mesh,
    load_point_cloud,
    normalize_point_cloud,
    sample_points_from_mesh,
)


def load_input_as_point_cloud(path: str, num_points: int = 2048, use_normals: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        vertices, faces = load_obj_mesh(path)
        points, normals = sample_points_from_mesh(vertices, faces, num_points=num_points, compute_normals=use_normals)
    elif ext == ".off":
        vertices, faces = load_off_mesh(path)
        points, normals = sample_points_from_mesh(vertices, faces, num_points=num_points, compute_normals=use_normals)
    else:
        points, normals = load_point_cloud(path)
        ids = farthest_point_sampling(points, num_points)
        points = points[ids]
        if normals is not None:
            normals = normals[ids]
        elif use_normals:
            normals = estimate_normals(points)
    points = normalize_point_cloud(points)
    return points.astype(np.float32), None if normals is None else normals.astype(np.float32)


def random_rotation(points: torch.Tensor) -> torch.Tensor:
    theta = torch.rand(points.shape[0], device=points.device) * 2.0 * np.pi
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    rot = torch.stack(
        [
            torch.stack([cos_t, -sin_t, torch.zeros_like(theta)], dim=-1),
            torch.stack([sin_t, cos_t, torch.zeros_like(theta)], dim=-1),
            torch.stack([torch.zeros_like(theta), torch.zeros_like(theta), torch.ones_like(theta)], dim=-1),
        ],
        dim=1,
    )
    return points @ rot.transpose(1, 2)


def jitter_points(points: torch.Tensor, sigma: float = 0.01, clip: float = 0.03) -> torch.Tensor:
    noise = torch.clamp(torch.randn_like(points) * sigma, min=-clip, max=clip)
    return points + noise


def make_augmented_features(features: torch.Tensor) -> torch.Tensor:
    points = features[:, :, :3]
    aug_points = jitter_points(random_rotation(points))
    if features.shape[-1] > 3:
        normals = features[:, :, 3:]
        aug_features = torch.cat([aug_points, normals], dim=-1)
    else:
        aug_features = aug_points
    return aug_features


def torch_kmeans(
    points: torch.Tensor,
    num_clusters: int,
    num_iters: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_points = points.shape[0]
    initial_ids = torch.randperm(num_points, device=points.device)[:num_clusters]
    centers = points[initial_ids]
    labels = torch.zeros(num_points, dtype=torch.long, device=points.device)
    for _ in range(num_iters):
        distances = torch.cdist(points, centers)
        labels = distances.argmin(dim=1)
        next_centers = []
        for idx in range(num_clusters):
            mask = labels == idx
            if mask.any():
                next_centers.append(points[mask].mean(dim=0))
            else:
                next_centers.append(centers[idx])
        centers = torch.stack(next_centers, dim=0)
    return labels, centers


def labels_to_colors(labels: np.ndarray) -> np.ndarray:
    palette = np.array(
        [
            [231, 76, 60],
            [52, 152, 219],
            [46, 204, 113],
            [241, 196, 15],
            [155, 89, 182],
            [230, 126, 34],
            [26, 188, 156],
            [149, 165, 166],
            [52, 73, 94],
            [243, 156, 18],
            [127, 140, 141],
            [192, 57, 43],
        ],
        dtype=np.uint8,
    )
    return palette[labels % len(palette)]


def save_colored_ply(path: str, points: np.ndarray, labels: np.ndarray) -> None:
    colors = labels_to_colors(labels)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def maybe_visualize(points: np.ndarray, labels: np.ndarray) -> None:
    try:
        import open3d as o3d
    except ImportError:
        return
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(labels_to_colors(labels).astype(np.float32) / 255.0)
    o3d.visualization.draw_geometries([cloud])
