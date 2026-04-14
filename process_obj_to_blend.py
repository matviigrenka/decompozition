import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full OBJ -> segmentation -> Blender .blend pipeline.")
    parser.add_argument("--input", required=True, help="Path to input .obj mesh")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint .pt")
    parser.add_argument("--blender", required=True, help="Path to Blender executable")
    parser.add_argument("--output-blend", required=True, help="Path to output .blend file")
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--num-clusters", type=int, default=6)
    parser.add_argument("--use-normals", action="store_true")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--use-slot-attention", action="store_true")
    parser.add_argument("--num-slots", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--separate", action="store_true", help="Split result into separate objects by cluster")
    parser.add_argument("--vertex-groups", action="store_true", help="Create Blender vertex groups for each cluster")
    parser.add_argument("--keep-intermediate", action="store_true", help="Keep intermediate PLY and NPZ outputs")
    parser.add_argument("--work-dir", default=None, help="Optional directory for intermediate files")
    return parser.parse_args()


def run_command(command):
    print("Running:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    input_path = Path(args.input).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    blender_path = Path(args.blender).resolve()
    output_blend = Path(args.output_blend).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input OBJ not found: {input_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not blender_path.exists():
        raise FileNotFoundError(f"Blender executable not found: {blender_path}")

    work_dir = Path(args.work_dir).resolve() if args.work_dir else output_blend.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    base_name = output_blend.stem
    ply_path = work_dir / f"{base_name}_segmented.ply"
    seg_npz_path = work_dir / f"{base_name}_segmentation.npz"

    inference_script = project_root / "inference.py"
    blender_script = project_root / "blender_decompose.py"

    inference_cmd = [
        sys.executable,
        str(inference_script),
        "--input",
        str(input_path),
        "--checkpoint",
        str(checkpoint_path),
        "--num-points",
        str(args.num_points),
        "--num-clusters",
        str(args.num_clusters),
        "--embedding-dim",
        str(args.embedding_dim),
        "--num-slots",
        str(args.num_slots),
        "--device",
        args.device,
        "--output",
        str(ply_path),
        "--blender-output",
        str(seg_npz_path),
    ]
    if args.use_normals:
        inference_cmd.append("--use-normals")
    if args.use_slot_attention:
        inference_cmd.append("--use-slot-attention")

    run_command(inference_cmd)

    blender_cmd = [
        str(blender_path),
        "-b",
        "--python",
        str(blender_script),
        "--",
        "--seg",
        str(seg_npz_path),
        "--mesh",
        str(input_path),
        "--output-blend",
        str(output_blend),
    ]
    if args.vertex_groups:
        blender_cmd.append("--vertex-groups")
    if args.separate:
        blender_cmd.append("--separate")

    run_command(blender_cmd)

    if not args.keep_intermediate:
        for path in (ply_path, seg_npz_path):
            if path.exists():
                path.unlink()

    print(f"Saved processed Blender file to {output_blend}")


if __name__ == "__main__":
    main()
