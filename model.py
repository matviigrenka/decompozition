from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    return torch.cdist(src, dst, p=2) ** 2


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    batch_indices = torch.arange(points.shape[0], device=points.device).view(-1, 1, 1)
    if idx.dim() == 2:
        batch_indices = batch_indices[:, :, 0]
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    device = xyz.device
    batch_size, num_points, _ = xyz.shape
    centroids = torch.zeros(batch_size, npoint, dtype=torch.long, device=device)
    distance = torch.full((batch_size, num_points), 1e10, device=device)
    farthest = torch.randint(0, num_points, (batch_size,), device=device)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def knn_point(k: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    dist = square_distance(new_xyz, xyz)
    k = min(k, xyz.shape[1])
    _, idx = torch.topk(dist, k=k, dim=-1, largest=False, sorted=False)
    return idx


def three_interpolate(src_xyz: torch.Tensor, dst_xyz: torch.Tensor, src_features: torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(dst_xyz, src_xyz, p=2)
    dists, idx = torch.topk(dist, k=min(3, src_xyz.shape[1]), dim=-1, largest=False)
    weights = 1.0 / (dists + 1e-8)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    gathered = index_points(src_features.transpose(1, 2), idx)
    interpolated = torch.sum(gathered * weights.unsqueeze(-1), dim=2)
    return interpolated.transpose(1, 2)


class SharedMLP(nn.Module):
    def __init__(self, channels):
        super().__init__()
        layers = []
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedMLP1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        layers = []
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SetAbstraction(nn.Module):
    def __init__(self, npoint: int, kneighbors: int, in_channels: int, mlp_channels):
        super().__init__()
        self.npoint = npoint
        self.kneighbors = kneighbors
        self.mlp = SharedMLP([in_channels + 3] + list(mlp_channels))

    def forward(
        self,
        xyz: torch.Tensor,
        features: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)
        group_idx = knn_point(self.kneighbors, xyz, new_xyz)
        grouped_xyz = index_points(xyz, group_idx)
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

        if features is None:
            grouped_features = grouped_xyz_norm
        else:
            feat_points = features.transpose(1, 2)
            grouped_features = index_points(feat_points, group_idx)
            grouped_features = torch.cat([grouped_xyz_norm, grouped_features], dim=-1)
        grouped_features = grouped_features.permute(0, 3, 1, 2)
        local_features = self.mlp(grouped_features)
        new_features = torch.max(local_features, dim=-1)[0]
        return new_xyz, new_features


class FeaturePropagation(nn.Module):
    def __init__(self, in_channels: int, mlp_channels):
        super().__init__()
        self.mlp = SharedMLP1d([in_channels] + list(mlp_channels))

    def forward(
        self,
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
        features1: Optional[torch.Tensor],
        features2: torch.Tensor,
    ) -> torch.Tensor:
        interpolated = three_interpolate(xyz2, xyz1, features2)
        if features1 is not None:
            fused = torch.cat([features1, interpolated], dim=1)
        else:
            fused = interpolated
        return self.mlp(fused)


class SlotAttention(nn.Module):
    def __init__(self, num_slots: int, dim: int, iters: int = 3) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)
        self.slots_mu = nn.Parameter(torch.zeros(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.ones(1, 1, dim))
        self.project_q = nn.Linear(dim, dim)
        self.project_k = nn.Linear(dim, dim)
        self.project_v = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(inplace=True), nn.Linear(dim * 2, dim))

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = inputs.shape[0]
        inputs = self.norm_inputs(inputs)
        mu = self.slots_mu.expand(batch_size, self.num_slots, -1)
        sigma = F.softplus(self.slots_sigma).expand(batch_size, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        k = self.project_k(inputs)
        v = self.project_v(inputs)
        scale = self.dim ** -0.5
        for _ in range(self.iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            q = self.project_q(slots_norm)
            attn_logits = torch.einsum("bsd,bnd->bsn", q, k) * scale
            attn = torch.softmax(attn_logits, dim=1) + 1e-8
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum("bsn,bnd->bsd", attn, v)
            slots = self.gru(
                updates.reshape(-1, self.dim),
                slots_prev.reshape(-1, self.dim),
            ).reshape(batch_size, self.num_slots, self.dim)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots, attn.transpose(1, 2)


class PointDecompositionModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        embedding_dim: int = 64,
        use_slot_attention: bool = False,
        num_slots: int = 8,
    ) -> None:
        super().__init__()
        self.use_slot_attention = use_slot_attention
        extra_channels = max(0, input_dim - 3)
        self.sa1 = SetAbstraction(npoint=512, kneighbors=32, in_channels=extra_channels, mlp_channels=[64, 64, 128])
        self.sa2 = SetAbstraction(npoint=128, kneighbors=32, in_channels=128, mlp_channels=[128, 128, 256])
        self.fp2 = FeaturePropagation(in_channels=128 + 256, mlp_channels=[256, 128])
        self.fp1 = FeaturePropagation(in_channels=extra_channels + 128, mlp_channels=[128, 128, 128])
        self.embedding_head = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, embedding_dim, kernel_size=1),
        )
        self.slot_attention = SlotAttention(num_slots=num_slots, dim=embedding_dim) if use_slot_attention else None

    def forward(self, x: torch.Tensor) -> dict:
        xyz = x[:, :, :3]
        extra = x[:, :, 3:] if x.shape[-1] > 3 else None
        extra = extra.transpose(1, 2) if extra is not None and extra.shape[-1] > 0 else None

        l1_xyz, l1_features = self.sa1(xyz, extra)
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
        l1_up = self.fp2(l1_xyz, l2_xyz, l1_features, l2_features)
        l0_up = self.fp1(xyz, l1_xyz, extra, l1_up)
        embeddings = self.embedding_head(l0_up).transpose(1, 2)
        embeddings = F.normalize(embeddings, dim=-1)

        output = {"embeddings": embeddings}
        if self.slot_attention is not None:
            slots, assignments = self.slot_attention(embeddings)
            slot_reconstruction = assignments @ slots
            refined = F.normalize(embeddings + slot_reconstruction, dim=-1)
            output["embeddings"] = refined
            output["slots"] = slots
            output["slot_assignments"] = assignments
        return output
