import os
import random
from typing import Dict, Optional, Tuple

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


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def prepare_inference_data(path: str, num_points: int = 2048, use_normals: bool = False) -> Dict[str, np.ndarray]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        vertices, faces = load_obj_mesh(path)
        original_points, normals = sample_points_from_mesh(vertices, faces, num_points=num_points, compute_normals=use_normals)
    elif ext == ".off":
        vertices, faces = load_off_mesh(path)
        original_points, normals = sample_points_from_mesh(vertices, faces, num_points=num_points, compute_normals=use_normals)
    else:
        points, normals = load_point_cloud(path)
        ids = farthest_point_sampling(points, num_points)
        original_points = points[ids]
        if normals is not None:
            normals = normals[ids]
        elif use_normals:
            normals = estimate_normals(original_points)

    normalized_points = normalize_point_cloud(original_points)
    if use_normals and normals is not None:
        features = np.concatenate([normalized_points, normals], axis=1)
    else:
        features = normalized_points
    return {
        "original_points": original_points.astype(np.float32),
        "normalized_points": normalized_points.astype(np.float32),
        "normals": None if normals is None else normals.astype(np.float32),
        "features": features.astype(np.float32),
    }


def load_input_as_point_cloud(path: str, num_points: int = 2048, use_normals: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    data = prepare_inference_data(path, num_points=num_points, use_normals=use_normals)
    return data["normalized_points"], data["normals"]


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


def _kmeans_plus_plus_init(
    points: torch.Tensor,
    num_clusters: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    num_points = points.shape[0]
    centers = torch.empty((num_clusters, points.shape[1]), device=points.device, dtype=points.dtype)
    first_idx = torch.randint(num_points, (1,), device=points.device, generator=generator).item()
    centers[0] = points[first_idx]
    closest_dist_sq = torch.sum((points - centers[0]) ** 2, dim=1)
    for center_idx in range(1, num_clusters):
        probs = closest_dist_sq / closest_dist_sq.sum().clamp_min(1e-12)
        next_idx = torch.multinomial(probs, num_samples=1, replacement=False, generator=generator).item()
        centers[center_idx] = points[next_idx]
        dist_sq = torch.sum((points - centers[center_idx]) ** 2, dim=1)
        closest_dist_sq = torch.minimum(closest_dist_sq, dist_sq)
    return centers


def torch_kmeans(
    points: torch.Tensor,
    num_clusters: int,
    num_iters: int = 30,
    num_restarts: int = 5,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_points = points.shape[0]
    if num_clusters > num_points:
        raise ValueError(f"num_clusters={num_clusters} is larger than the number of points={num_points}")

    best_inertia = None
    best_labels = None
    best_centers = None
    generator = None
    if seed is not None:
        generator = torch.Generator(device=points.device)
        generator.manual_seed(seed)

    for _ in range(max(1, num_restarts)):
        centers = _kmeans_plus_plus_init(points, num_clusters, generator=generator)
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
            next_centers = torch.stack(next_centers, dim=0)
            if torch.allclose(next_centers, centers, atol=1e-5):
                centers = next_centers
                break
            centers = next_centers
        final_distances = torch.cdist(points, centers)
        inertia = final_distances[torch.arange(num_points, device=points.device), labels].pow(2).mean()
        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.clone()
            best_centers = centers.clone()

    return best_labels, best_centers


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


def save_blender_segmentation(path: str, input_path: str, sampled_points: np.ndarray, labels: np.ndarray) -> None:
    np.savez_compressed(
        path,
        input_path=np.asarray(input_path),
        sampled_points=sampled_points.astype(np.float32),
        labels=labels.astype(np.int32),
        colors=labels_to_colors(labels).astype(np.uint8),
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
