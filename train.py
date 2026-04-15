import argparse
import os

import torch
from torch.utils.data import DataLoader

from dataset import MeshPointCloudDataset
from loss import total_unsupervised_loss
from model import PointDecompositionModel
from utils import make_augmented_features, set_global_seed

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback keeps training usable without tqdm
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an unsupervised 3D part decomposition model.")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory with .obj/.off/.xyz/.txt/.pts/.npy/.npz files.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-clusters", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use-normals", action="store_true")
    parser.add_argument("--use-slot-attention", action="store_true")
    parser.add_argument("--num-slots", type=int, default=8)
    parser.add_argument("--synthetic-size", type=int, default=64, help="Used when no real dataset directory is provided.")
    parser.add_argument("--save-path", type=str, default="checkpoint.pt")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA.")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision even when CUDA is available.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested with --device cuda, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    device = resolve_device(args.device)
    use_cuda = device.type == "cuda"
    use_amp = use_cuda and not args.no_amp
    if args.amp:
        use_amp = use_cuda

    if use_cuda and not args.amp and not args.no_amp:
        print("AMP is available. Consider adding --amp for faster RTX inference/training.")

    if use_cuda:
        device_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {device_name}")
    else:
        print("Using CPU")

    dataset = MeshPointCloudDataset(
        data_dir=args.data_dir,
        num_points=args.num_points,
        use_normals=args.use_normals,
        synthetic_size=args.synthetic_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=args.num_workers > 0,
    )

    model = PointDecompositionModel(
        input_dim=6 if args.use_normals else 3,
        embedding_dim=args.embedding_dim,
        use_slot_attention=args.use_slot_attention,
        num_slots=args.num_slots,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    epoch_bar = make_progress(range(args.epochs), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        running = 0.0
        last_stats = None
        batch_bar = make_progress(
            loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            unit="batch",
            leave=False,
        )
        for batch in batch_bar:
            points = batch["points"].to(device, non_blocking=use_cuda)
            features = batch["features"].to(device, non_blocking=use_cuda)
            aug_features = make_augmented_features(features)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(features)
                outputs_aug = model(aug_features)
                loss, stats = total_unsupervised_loss(
                    points=points,
                    embeddings=outputs["embeddings"],
                    num_clusters=args.num_clusters,
                    contrastive_pair=outputs_aug["embeddings"],
                )

                if args.use_slot_attention and "slots" in outputs:
                    slot_separation = torch.pdist(outputs["slots"].reshape(-1, outputs["slots"].shape[-1])).mean()
                    loss = loss - 0.05 * slot_separation

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            last_stats = stats

            if tqdm is not None:
                batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        average = running / max(1, len(loader))
        if last_stats is None:
            print(f"epoch={epoch + 1}/{args.epochs} loss={average:.4f}")
        else:
            if tqdm is not None:
                epoch_bar.set_postfix(
                    loss=f"{average:.4f}",
                    smooth=f"{last_stats['smooth'].item():.4f}",
                    separation=f"{last_stats['separation'].item():.4f}",
                )
            print(
                f"epoch={epoch + 1}/{args.epochs} loss={average:.4f} "
                f"smooth={last_stats['smooth'].item():.4f} separation={last_stats['separation'].item():.4f} "
                f"compactness={last_stats['compactness'].item():.4f} contrastive={last_stats['contrastive'].item():.4f}"
            )

    checkpoint = {
        "model_state": model.state_dict(),
        "input_dim": 6 if args.use_normals else 3,
        "embedding_dim": args.embedding_dim,
        "num_slots": args.num_slots,
        "use_slot_attention": args.use_slot_attention,
        "num_clusters": args.num_clusters,
        "device": str(device),
        "amp": use_amp,
        "seed": args.seed,
    }
    torch.save(checkpoint, args.save_path)
    print(f"Saved checkpoint to {os.path.abspath(args.save_path)}")


if __name__ == "__main__":
    main()
