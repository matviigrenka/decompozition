from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _knn_indices(points: torch.Tensor, k: int) -> torch.Tensor:
    dists = torch.cdist(points, points)
    k = min(k + 1, points.shape[1])
    _, idx = torch.topk(dists, k=k, largest=False)
    return idx[:, :, 1:]


def spatial_smoothness_loss(points: torch.Tensor, embeddings: torch.Tensor, k: int = 16) -> torch.Tensor:
    idx = _knn_indices(points, k)
    batch_ids = torch.arange(embeddings.shape[0], device=embeddings.device).view(-1, 1, 1)
    neighbors = embeddings[batch_ids, idx, :]
    center = embeddings.unsqueeze(2)
    loss = ((center - neighbors) ** 2).sum(dim=-1).mean()
    return loss


def separation_loss(
    points: torch.Tensor,
    embeddings: torch.Tensor,
    margin: float = 1.0,
    far_ratio: float = 0.25,
) -> torch.Tensor:
    spatial_dist = torch.cdist(points, points)
    emb_dist = torch.cdist(embeddings, embeddings)
    threshold = torch.quantile(spatial_dist.detach(), 1.0 - far_ratio)
    far_mask = spatial_dist >= threshold
    far_mask = far_mask & (~torch.eye(points.shape[1], device=points.device, dtype=torch.bool).unsqueeze(0))
    if far_mask.sum() == 0:
        return torch.tensor(0.0, device=points.device)
    penalties = F.relu(margin - emb_dist) ** 2
    return penalties[far_mask].mean()


def _batch_kmeans(embeddings: torch.Tensor, num_clusters: int, iters: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_points, channels = embeddings.shape
    initial_ids = torch.randperm(num_points, device=embeddings.device)[:num_clusters]
    centers = embeddings[:, initial_ids, :].contiguous()
    assignments = None
    for _ in range(iters):
        distances = torch.cdist(embeddings, centers)
        assignments = distances.argmin(dim=-1)
        new_centers = []
        for cluster_idx in range(num_clusters):
            mask = assignments == cluster_idx
            cluster_mask = mask.unsqueeze(-1)
            count = cluster_mask.sum(dim=1).clamp_min(1)
            cluster_sum = (embeddings * cluster_mask).sum(dim=1)
            center = cluster_sum / count
            fallback = centers[:, cluster_idx, :]
            is_empty = (mask.sum(dim=1) == 0).unsqueeze(-1)
            center = torch.where(is_empty, fallback, center)
            new_centers.append(center)
        centers = torch.stack(new_centers, dim=1)
    return centers, assignments


def compactness_loss(embeddings: torch.Tensor, num_clusters: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    centers, assignments = _batch_kmeans(embeddings, num_clusters=num_clusters)
    selected_centers = centers.gather(1, assignments.unsqueeze(-1).expand(-1, -1, embeddings.shape[-1]))
    loss = ((embeddings - selected_centers) ** 2).sum(dim=-1).mean()
    return loss, assignments


def info_nce_loss(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    batch_size, num_points, channels = embeddings_a.shape
    flat_a = F.normalize(embeddings_a.reshape(batch_size * num_points, channels), dim=-1)
    flat_b = F.normalize(embeddings_b.reshape(batch_size * num_points, channels), dim=-1)
    logits = flat_a @ flat_b.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


def total_unsupervised_loss(
    points: torch.Tensor,
    embeddings: torch.Tensor,
    num_clusters: int,
    contrastive_pair: Optional[torch.Tensor] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    weights = weights or {
        "smooth": 1.0,
        "separation": 1.0,
        "compactness": 0.5,
        "contrastive": 0.1,
    }
    smooth = spatial_smoothness_loss(points, embeddings)
    separate = separation_loss(points, embeddings)
    compact, pseudo_labels = compactness_loss(embeddings, num_clusters=num_clusters)
    contrastive = (
        info_nce_loss(embeddings, contrastive_pair)
        if contrastive_pair is not None
        else torch.tensor(0.0, device=embeddings.device)
    )
    total = (
        weights["smooth"] * smooth
        + weights["separation"] * separate
        + weights["compactness"] * compact
        + weights["contrastive"] * contrastive
    )
    return total, {
        "smooth": smooth.detach(),
        "separation": separate.detach(),
        "compactness": compact.detach(),
        "contrastive": contrastive.detach(),
        "pseudo_labels": pseudo_labels.detach(),
    }
