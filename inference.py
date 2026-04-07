import argparse
import os

import numpy as np
import torch

from model import PointDecompositionModel
from utils import load_input_as_point_cloud, maybe_visualize, save_colored_ply, torch_kmeans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer unsupervised 3D parts from a mesh or point cloud.")
    parser.add_argument("--input", type=str, required=True, help="Path to .obj or point cloud file.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint created by train.py")
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--num-clusters", type=int, default=6)
    parser.add_argument("--use-normals", action="store_true")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--use-slot-attention", action="store_true")
    parser.add_argument("--num-slots", type=int, default=8)
    parser.add_argument("--output", type=str, default="segmented_output.ply")
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> PointDecompositionModel:
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model = PointDecompositionModel(
            input_dim=checkpoint.get("input_dim", 6 if args.use_normals else 3),
            embedding_dim=checkpoint.get("embedding_dim", args.embedding_dim),
            use_slot_attention=checkpoint.get("use_slot_attention", args.use_slot_attention),
            num_slots=checkpoint.get("num_slots", args.num_slots),
        )
        model.load_state_dict(checkpoint["model_state"])
        return model
    return PointDecompositionModel(
        input_dim=6 if args.use_normals else 3,
        embedding_dim=args.embedding_dim,
        use_slot_attention=args.use_slot_attention,
        num_slots=args.num_slots,
    )


def main() -> None:
    args = parse_args()
    points, normals = load_input_as_point_cloud(args.input, num_points=args.num_points, use_normals=args.use_normals)
    if args.use_normals and normals is not None:
        features = np.concatenate([points, normals], axis=1)
    else:
        features = points

    model = build_model(args)
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(features).unsqueeze(0)
        outputs = model(tensor)
        embeddings = outputs["embeddings"][0]
        labels, _ = torch_kmeans(embeddings, num_clusters=args.num_clusters)

    labels_np = labels.cpu().numpy().astype(np.int32)
    save_colored_ply(args.output, points, labels_np)
    print(f"Saved segmentation to {os.path.abspath(args.output)}")

    if args.visualize:
        maybe_visualize(points, labels_np)


if __name__ == "__main__":
    main()
