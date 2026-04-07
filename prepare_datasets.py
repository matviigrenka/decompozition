import argparse
import json
import os
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from dataset import (
    estimate_normals,
    farthest_point_sampling,
    load_obj_mesh,
    load_off_mesh,
    load_point_cloud,
    normalize_point_cloud,
    sample_points_from_mesh,
)


DATASET_SPECS: Dict[str, Dict[str, object]] = {
    "modelnet40": {
        "source": "official",
        "download_url": "http://modelnet.cs.princeton.edu/ModelNet40.zip",
        "archive_name": "ModelNet40.zip",
        "expected_root": "ModelNet40",
        "format": "modelnet",
        "notes": "Open dataset. Download is fully automatic.",
    },
    "modelnet10": {
        "source": "official",
        "download_url": "http://modelnet.cs.princeton.edu/ModelNet10.zip",
        "archive_name": "ModelNet10.zip",
        "expected_root": "ModelNet10",
        "format": "modelnet",
        "notes": "Open dataset. Download is fully automatic.",
    },
    "shapenetcore": {
        "source": "manual",
        "expected_root": "ShapeNetCore.v2",
        "format": "generic",
        "notes": "Register at https://www.shapenet.org/ and place the extracted folder under raw_datasets/shapenetcore/ShapeNetCore.v2",
    },
    "partnet": {
        "source": "manual",
        "expected_root": "PartNet",
        "format": "generic",
        "notes": "Request access at https://partnet.cs.stanford.edu/ and place the extracted folder under raw_datasets/partnet/PartNet",
    },
    "scanobjectnn": {
        "source": "manual",
        "expected_root": "ScanObjectNN",
        "format": "generic",
        "notes": "Download from https://hkust-vgd.github.io/scanobjectnn/ and place mesh/point files under raw_datasets/scanobjectnn/ScanObjectNN",
    },
    "custom": {
        "source": "manual",
        "expected_root": "custom",
        "format": "generic",
        "notes": "Put any .obj/.off/.xyz/.txt/.pts/.npy/.npz files under raw_datasets/custom/custom",
    },
}
SUPPORTED_EXTENSIONS = {".obj", ".off", ".xyz", ".txt", ".pts", ".npy", ".npz"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and convert 3D datasets into a unified NPZ point-cloud format.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="modelnet40",
        help="Comma-separated list: modelnet40,modelnet10,shapenetcore,partnet,scanobjectnn,custom",
    )
    parser.add_argument("--raw-root", type=str, default="raw_datasets")
    parser.add_argument("--out-root", type=str, default="unified_datasets")
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--with-normals", action="store_true")
    parser.add_argument("--redownload", action="store_true")
    parser.add_argument("--reprocess", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path) -> None:
    ensure_dir(destination.parent)
    with urllib.request.urlopen(url) as response, open(destination, "wb") as handle:
        total = response.headers.get("Content-Length")
        total_size = int(total) if total is not None else None
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if total_size:
                pct = 100.0 * downloaded / total_size
                print(f"  downloaded {downloaded / (1024 * 1024):.1f} MB / {total_size / (1024 * 1024):.1f} MB ({pct:.1f}%)", end="\r")
    print()


def extract_archive(archive_path: Path, destination: Path) -> Path:
    ensure_dir(destination)
    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(destination)
    elif archive_path.suffix.lower() in {".gz", ".tgz"} or archive_path.name.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(destination)
    elif archive_path.suffix.lower() == ".tar":
        with tarfile.open(archive_path, "r") as archive:
            archive.extractall(destination)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")
    return destination


def maybe_download_dataset(name: str, raw_root: Path, redownload: bool, dry_run: bool) -> Path:
    spec = DATASET_SPECS[name]
    dataset_root = raw_root / name
    ensure_dir(dataset_root)
    expected_root = dataset_root / str(spec["expected_root"])
    if spec["source"] == "manual":
        print(f"[{name}] manual dataset: {spec['notes']}")
        return expected_root

    archive_path = dataset_root / str(spec["archive_name"])
    if archive_path.exists() and not redownload:
        print(f"[{name}] using existing archive: {archive_path}")
    else:
        print(f"[{name}] downloading from {spec['download_url']}")
        if dry_run:
            print(f"[{name}] dry-run: skipping download")
        else:
            download_file(str(spec["download_url"]), archive_path)

    if expected_root.exists() and any(expected_root.iterdir()) and not redownload:
        print(f"[{name}] using extracted directory: {expected_root}")
    else:
        if dry_run:
            print(f"[{name}] dry-run: skipping extraction")
        else:
            if expected_root.exists():
                shutil.rmtree(expected_root)
            print(f"[{name}] extracting {archive_path.name}")
            extract_archive(archive_path, dataset_root)
    return expected_root


def load_geometry(path: Path, num_points: int, with_normals: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    ext = path.suffix.lower()
    if ext == ".obj":
        vertices, faces = load_obj_mesh(str(path))
        points, normals = sample_points_from_mesh(vertices, faces, num_points=num_points, compute_normals=with_normals)
    elif ext == ".off":
        vertices, faces = load_off_mesh(str(path))
        points, normals = sample_points_from_mesh(vertices, faces, num_points=num_points, compute_normals=with_normals)
    else:
        points, normals = load_point_cloud(str(path))
        ids = farthest_point_sampling(points, num_points)
        points = points[ids]
        if normals is not None:
            normals = normals[ids]
        elif with_normals:
            normals = estimate_normals(points)
    points = normalize_point_cloud(points)
    return points.astype(np.float32), None if normals is None else normals.astype(np.float32)


def infer_split_and_category(dataset_name: str, source_path: Path, root: Path) -> Tuple[str, str]:
    rel_parts = source_path.relative_to(root).parts
    split = "unspecified"
    category = dataset_name
    split_tokens = {"train", "test", "val", "valid", "validation"}
    for token in rel_parts:
        lowered = token.lower()
        if lowered in split_tokens:
            split = "val" if lowered in {"valid", "validation"} else lowered
            break
    if dataset_name.startswith("modelnet") and len(rel_parts) >= 2:
        category = rel_parts[0]
    elif len(rel_parts) >= 2:
        category = rel_parts[-2]
    return split, category


def iter_geometry_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def save_sample(
    out_path: Path,
    points: np.ndarray,
    normals: Optional[np.ndarray],
    metadata: Dict[str, str],
) -> None:
    ensure_dir(out_path.parent)
    if normals is None:
        np.savez_compressed(out_path, points=points, normals=np.zeros((0, 3), dtype=np.float32), **metadata)
    else:
        np.savez_compressed(out_path, points=points, normals=normals, **metadata)


def process_dataset(
    dataset_name: str,
    source_root: Path,
    out_root: Path,
    num_points: int,
    with_normals: bool,
    reprocess: bool,
    dry_run: bool,
) -> None:
    if not source_root.exists():
        print(f"[{dataset_name}] source directory not found: {source_root}")
        return

    target_root = out_root / dataset_name
    ensure_dir(target_root)
    manifest_path = target_root / "manifest.jsonl"
    existing_manifest = set()
    if manifest_path.exists() and not reprocess:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                existing_manifest.add(record["output_path"])

    records: List[Dict[str, str]] = []
    processed = 0
    skipped = 0
    for source_path in iter_geometry_files(source_root):
        split, category = infer_split_and_category(dataset_name, source_path, source_root)
        sample_id = source_path.stem
        rel_id = str(Path(split) / category / f"{sample_id}.npz")
        out_path = target_root / rel_id
        if rel_id in existing_manifest and out_path.exists() and not reprocess:
            skipped += 1
            continue
        print(f"[{dataset_name}] processing {source_path}")
        if dry_run:
            continue
        try:
            points, normals = load_geometry(source_path, num_points=num_points, with_normals=with_normals)
        except Exception as exc:
            print(f"[{dataset_name}] skipped {source_path} due to error: {exc}")
            continue
        metadata = {
            "dataset": dataset_name,
            "split": split,
            "category": category,
            "source_path": str(source_path),
        }
        save_sample(out_path, points, normals, metadata)
        records.append(
            {
                "dataset": dataset_name,
                "split": split,
                "category": category,
                "source_path": str(source_path),
                "output_path": rel_id,
                "num_points": int(points.shape[0]),
                "has_normals": bool(normals is not None),
            }
        )
        processed += 1

    if dry_run:
        print(f"[{dataset_name}] dry-run complete")
        return

    if reprocess and manifest_path.exists():
        manifest_path.unlink()
    if records:
        with open(manifest_path, "a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
    print(f"[{dataset_name}] done: processed={processed}, skipped={skipped}, manifest={manifest_path}")


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    ensure_dir(raw_root)
    ensure_dir(out_root)

    dataset_names = [name.strip().lower() for name in args.datasets.split(",") if name.strip()]
    unknown = [name for name in dataset_names if name not in DATASET_SPECS]
    if unknown:
        raise ValueError(f"Unknown datasets requested: {unknown}. Available: {sorted(DATASET_SPECS)}")

    for dataset_name in dataset_names:
        source_root = maybe_download_dataset(dataset_name, raw_root, args.redownload, args.dry_run)
        process_dataset(
            dataset_name=dataset_name,
            source_root=source_root,
            out_root=out_root,
            num_points=args.num_points,
            with_normals=args.with_normals,
            reprocess=args.reprocess,
            dry_run=args.dry_run,
        )

    print("Unified dataset preparation finished.")


if __name__ == "__main__":
    main()
