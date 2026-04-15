import argparse
import os

import torch

from model import PointDecompositionModel
from utils import maybe_visualize, prepare_inference_data, save_blender_segmentation, save_colored_ply, set_global_seed, torch_kmeans


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
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kmeans-iters", type=int, default=30)
    parser.add_argument("--kmeans-restarts", type=int, default=5)
    parser.add_argument("--output", type=str, default="segmented_output.ply")
    parser.add_argument(
        "--blender-output",
        type=str,
        default=None,
        help="Optional NPZ file for Blender projection. Defaults to <output>_blender.npz",
    )
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested with --device cuda, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint_metadata(args: argparse.Namespace, device: torch.device):
    if args.checkpoint is None:
        return None
    return torch.load(args.checkpoint, map_location=device)


def build_model(args: argparse.Namespace, device: torch.device, checkpoint=None) -> PointDecompositionModel:
    if checkpoint is not None:
        model = PointDecompositionModel(
            input_dim=checkpoint.get("input_dim", 6 if args.use_normals else 3),
            embedding_dim=checkpoint.get("embedding_dim", args.embedding_dim),
            use_slot_attention=checkpoint.get("use_slot_attention", args.use_slot_attention),
            num_slots=checkpoint.get("num_slots", args.num_slots),
        )
        model.load_state_dict(checkpoint["model_state"])
        return model.to(device)
    return PointDecompositionModel(
        input_dim=6 if args.use_normals else 3,
        embedding_dim=args.embedding_dim,
        use_slot_attention=args.use_slot_attention,
        num_slots=args.num_slots,
    ).to(device)


def default_blender_output(output_path: str) -> str:
    root, _ = os.path.splitext(output_path)
    return root + "_blender.npz"


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint = load_checkpoint_metadata(args, device)

    expected_input_dim = checkpoint.get("input_dim", 6 if args.use_normals else 3) if checkpoint is not None else (6 if args.use_normals else 3)
    effective_use_normals = expected_input_dim > 3
    if args.use_normals and not effective_use_normals:
        print("Checkpoint was trained without normals (input_dim=3). Ignoring --use-normals for inference.")
    if not args.use_normals and effective_use_normals:
        print("Checkpoint expects normals (input_dim=6). Computing normals automatically for inference.")

    inference_data = prepare_inference_data(args.input, num_points=args.num_points, use_normals=effective_use_normals)
    points = inference_data["normalized_points"]
    features = inference_data["features"]

    model = build_model(args, device, checkpoint=checkpoint)
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(features).unsqueeze(0).to(device)
        outputs = model(tensor)
        embeddings = outputs["embeddings"][0]
        labels, _ = torch_kmeans(
            embeddings,
            num_clusters=args.num_clusters,
            num_iters=args.kmeans_iters,
            num_restarts=args.kmeans_restarts,
            seed=args.seed,
        )

    labels_np = labels.cpu().numpy().astype("int32")
    save_colored_ply(args.output, points, labels_np)
    print(f"Saved segmentation point cloud to {os.path.abspath(args.output)}")

    blender_output = args.blender_output or default_blender_output(args.output)
    save_blender_segmentation(
        blender_output,
        input_path=args.input,
        sampled_points=inference_data["original_points"],
        labels=labels_np,
    )
    print(f"Saved Blender projection data to {os.path.abspath(blender_output)}")

    if args.visualize:
        maybe_visualize(points, labels_np)


if __name__ == "__main__":
    main()
