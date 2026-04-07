import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def load_obj_mesh(path: str) -> Tuple[np.ndarray, np.ndarray]:
    vertices: List[List[float]] = []
    faces: List[List[int]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            tokens = line.strip().split()
            if tokens[0] == "v" and len(tokens) >= 4:
                vertices.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
            elif tokens[0] == "f" and len(tokens) >= 4:
                face = []
                for item in tokens[1:]:
                    index = item.split("/")[0]
                    if index:
                        face.append(int(index) - 1)
                if len(face) >= 3:
                    for i in range(1, len(face) - 1):
                        faces.append([face[0], face[i], face[i + 1]])
    if not vertices:
        raise ValueError(f"No vertices found in OBJ file: {path}")
    vertices_np = np.asarray(vertices, dtype=np.float32)
    faces_np = np.asarray(faces, dtype=np.int64) if faces else np.zeros((0, 3), dtype=np.int64)
    return vertices_np, faces_np


def load_off_mesh(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline().strip()
        if header == "OFF":
            counts_line = handle.readline().strip()
        elif header.startswith("OFF"):
            counts_line = header[3:].strip()
        else:
            raise ValueError(f"Invalid OFF header in file: {path}")
        while not counts_line or counts_line.startswith("#"):
            counts_line = handle.readline().strip()
        num_vertices, num_faces, _ = map(int, counts_line.split()[:3])
        vertices = []
        for _ in range(num_vertices):
            values = handle.readline().strip().split()
            vertices.append([float(values[0]), float(values[1]), float(values[2])])
        faces = []
        for _ in range(num_faces):
            tokens = handle.readline().strip().split()
            if not tokens:
                continue
            degree = int(tokens[0])
            indices = [int(value) for value in tokens[1 : 1 + degree]]
            if len(indices) >= 3:
                for idx in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[idx], indices[idx + 1]])
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def load_point_cloud(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        array = np.load(path)
    elif ext == ".npz":
        data = np.load(path)
        if "points" not in data:
            raise ValueError(f"Unified NPZ file is missing 'points': {path}")
        points = data["points"].astype(np.float32)
        normals = data["normals"].astype(np.float32) if "normals" in data and data["normals"].size > 0 else None
        return points, normals
    else:
        array = np.loadtxt(path, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError(f"Expected point cloud with shape Nx3 or Nx6, got {array.shape}")
    points = array[:, :3].astype(np.float32)
    normals = array[:, 3:6].astype(np.float32) if array.shape[1] >= 6 else None
    return points, normals


def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(centered, axis=1).max()
    if scale < 1e-8:
        return centered
    return centered / scale


def estimate_normals(points: np.ndarray, k: int = 16) -> np.ndarray:
    diffs = points[:, None, :] - points[None, :, :]
    dists = np.sum(diffs * diffs, axis=-1)
    neighbor_ids = np.argsort(dists, axis=1)[:, 1 : k + 1]
    normals = np.zeros_like(points)
    for idx in range(points.shape[0]):
        neighbors = points[neighbor_ids[idx]]
        centered = neighbors - neighbors.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(1, centered.shape[0])
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normal = eigenvectors[:, np.argmin(eigenvalues)]
        normals[idx] = normal / (np.linalg.norm(normal) + 1e-8)
    return normals.astype(np.float32)


def sample_points_from_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_points: int,
    compute_normals: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if faces.shape[0] == 0:
        ids = np.random.choice(vertices.shape[0], size=num_points, replace=vertices.shape[0] < num_points)
        points = vertices[ids]
        normals = estimate_normals(points) if compute_normals else None
        return points.astype(np.float32), normals

    tri_vertices = vertices[faces]
    edge1 = tri_vertices[:, 1] - tri_vertices[:, 0]
    edge2 = tri_vertices[:, 2] - tri_vertices[:, 0]
    face_normals = np.cross(edge1, edge2)
    face_areas = 0.5 * np.linalg.norm(face_normals, axis=1)
    area_sum = face_areas.sum()
    if area_sum < 1e-8:
        ids = np.random.choice(vertices.shape[0], size=num_points, replace=vertices.shape[0] < num_points)
        points = vertices[ids]
        normals = estimate_normals(points) if compute_normals else None
        return points.astype(np.float32), normals

    probs = face_areas / area_sum
    chosen_faces = np.random.choice(faces.shape[0], size=num_points, p=probs)
    picked = tri_vertices[chosen_faces]

    u = np.random.rand(num_points, 1).astype(np.float32)
    v = np.random.rand(num_points, 1).astype(np.float32)
    mask = (u + v) > 1.0
    u[mask] = 1.0 - u[mask]
    v[mask] = 1.0 - v[mask]
    w = 1.0 - (u + v)
    points = picked[:, 0] * u + picked[:, 1] * v + picked[:, 2] * w

    normals = None
    if compute_normals:
        chosen_normals = face_normals[chosen_faces]
        denom = np.linalg.norm(chosen_normals, axis=1, keepdims=True) + 1e-8
        normals = (chosen_normals / denom).astype(np.float32)
    return points.astype(np.float32), normals


def farthest_point_sampling(points: np.ndarray, num_samples: int) -> np.ndarray:
    if points.shape[0] <= num_samples:
        if points.shape[0] < num_samples:
            extra = np.random.choice(points.shape[0], num_samples - points.shape[0], replace=True)
            return np.concatenate([np.arange(points.shape[0]), extra])
        return np.arange(points.shape[0])
    indices = np.zeros(num_samples, dtype=np.int64)
    distances = np.full(points.shape[0], 1e10, dtype=np.float32)
    farthest = np.random.randint(0, points.shape[0])
    for i in range(num_samples):
        indices[i] = farthest
        centroid = points[farthest]
        dist = np.sum((points - centroid) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest = int(np.argmax(distances))
    return indices


class MeshPointCloudDataset(Dataset):
    def __init__(
        self,
        data_dir: Optional[str] = None,
        num_points: int = 2048,
        use_normals: bool = False,
        synthetic_size: int = 128,
        normalize: bool = True,
    ) -> None:
        self.data_dir = data_dir
        self.num_points = num_points
        self.use_normals = use_normals
        self.synthetic_size = synthetic_size
        self.normalize = normalize
        self.files = self._discover_files(data_dir)

    def _discover_files(self, data_dir: Optional[str]) -> List[str]:
        if data_dir is None or not os.path.isdir(data_dir):
            return []
        supported = {".obj", ".off", ".xyz", ".txt", ".pts", ".npy", ".npz"}
        files = []
        for root, _, names in os.walk(data_dir):
            for name in names:
                path = os.path.join(root, name)
                if os.path.isfile(path) and os.path.splitext(name)[1].lower() in supported:
                    files.append(path)
        files.sort()
        return files

    def __len__(self) -> int:
        return len(self.files) if self.files else self.synthetic_size

    def _load_real_sample(self, path: str) -> Dict[str, torch.Tensor]:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".obj":
            vertices, faces = load_obj_mesh(path)
            points, normals = sample_points_from_mesh(vertices, faces, self.num_points, self.use_normals)
        elif ext == ".off":
            vertices, faces = load_off_mesh(path)
            points, normals = sample_points_from_mesh(vertices, faces, self.num_points, self.use_normals)
        else:
            points, normals = load_point_cloud(path)
            ids = farthest_point_sampling(points, self.num_points)
            points = points[ids]
            if normals is not None:
                normals = normals[ids]
            elif self.use_normals:
                normals = estimate_normals(points)
        if self.normalize:
            points = normalize_point_cloud(points)
        features = points if not self.use_normals else np.concatenate([points, normals], axis=1)
        return {
            "points": torch.from_numpy(points.astype(np.float32)),
            "features": torch.from_numpy(features.astype(np.float32)),
            "path": path,
        }

    def _sample_primitive(self) -> np.ndarray:
        primitive = np.random.choice(["sphere", "cuboid", "cylinder"])
        n = self.num_points
        if primitive == "sphere":
            phi = np.random.rand(n) * 2.0 * math.pi
            costheta = np.random.rand(n) * 2.0 - 1.0
            sintheta = np.sqrt(1.0 - costheta ** 2)
            radius = 0.15 + 0.12 * np.random.rand()
            center = np.random.uniform(-0.5, 0.5, size=3)
            pts = np.stack([
                radius * sintheta * np.cos(phi),
                radius * sintheta * np.sin(phi),
                radius * costheta,
            ], axis=1)
            return (pts + center).astype(np.float32)
        if primitive == "cuboid":
            extents = np.random.uniform(0.08, 0.24, size=3)
            center = np.random.uniform(-0.45, 0.45, size=3)
            face_ids = np.random.randint(0, 6, size=n)
            pts = np.random.uniform(-1.0, 1.0, size=(n, 3)) * extents
            axis = face_ids // 2
            sign = (face_ids % 2) * 2 - 1
            pts[np.arange(n), axis] = sign * extents[axis]
            return (pts + center).astype(np.float32)
        radius = 0.08 + 0.08 * np.random.rand()
        height = 0.25 + 0.25 * np.random.rand()
        center = np.random.uniform(-0.45, 0.45, size=3)
        theta = np.random.rand(n) * 2.0 * math.pi
        z = np.random.rand(n) * height - height / 2.0
        pts = np.stack([radius * np.cos(theta), radius * np.sin(theta), z], axis=1)
        return (pts + center).astype(np.float32)

    def _generate_synthetic_object(self) -> Dict[str, torch.Tensor]:
        num_parts = np.random.randint(2, 6)
        base_points = []
        for _ in range(num_parts):
            base_points.append(self._sample_primitive())
        points = np.concatenate(base_points, axis=0)
        ids = farthest_point_sampling(points, self.num_points)
        points = points[ids]
        if self.normalize:
            points = normalize_point_cloud(points)
        normals = estimate_normals(points) if self.use_normals else None
        features = points if normals is None else np.concatenate([points, normals], axis=1)
        return {
            "points": torch.from_numpy(points.astype(np.float32)),
            "features": torch.from_numpy(features.astype(np.float32)),
            "path": "synthetic",
        }

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self.files:
            return self._load_real_sample(self.files[index])
        return self._generate_synthetic_object()
