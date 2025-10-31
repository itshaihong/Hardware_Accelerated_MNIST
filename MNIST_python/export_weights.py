import os
import json
import argparse
from typing import Dict, Any

import torch
import numpy as np

from lenet5 import LeNet5

DEFAULT_FMT = "%.7g"  # human-readable; adjust if you want more precision


def save_tensor_csv(tensor: torch.Tensor, path: str, flatten: bool, fmt: str = DEFAULT_FMT):
    arr = tensor.detach().cpu().numpy()
    if flatten:
        arr = arr.reshape(-1)  # 1D, row-major (C-contiguous) flatten
        np.savetxt(path, arr, fmt=fmt, delimiter=",")
    else:
        # For 1D: single row; For 2D+: we try to save with rows preserved where possible.
        # For >2D arrays, we'll reshape to (first_dim, -1) to keep top-level slices as rows.
        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim >= 3:
            first = arr.shape[0]
            arr = arr.reshape(first, -1)
        np.savetxt(path, arr, fmt=fmt, delimiter=",")


def export_state_dict(
    state_dict: Dict[str, torch.Tensor],
    out_dir: str,
    flatten: bool,
    fmt: str = DEFAULT_FMT
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "format": "csv",
        "precision": fmt,
        "flatten": flatten,
        "tensors": []
    }

    for name, tensor in state_dict.items():
        safe_name = name.replace(".", "_")  # e.g., conv1.weight -> conv1_weight
        csv_path = os.path.join(out_dir, f"{safe_name}.csv")
        save_tensor_csv(tensor, csv_path, flatten=flatten, fmt=fmt)

        entry = {
            "name": name,
            "file": os.path.basename(csv_path),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", "")
        }
        manifest["tensors"].append(entry)

    # Save a manifest JSON to document shapes and file mapping
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Export LeNet-5 PyTorch weights to CSV")
    parser.add_argument("--model", type=str, default="weights/lenet5.pth", help="Path to model .pth")
    parser.add_argument("--out_dir", type=str, default="weights_csv", help="Directory to write CSV files")
    parser.add_argument("--flatten", action="store_true", help="Flatten tensors to 1D CSV")
    parser.add_argument("--fmt", type=str, default=DEFAULT_FMT, help="NumPy print format (e.g., %.7g, %.10f)")
    args = parser.parse_args()

    # Load model and weights
    model = LeNet5(num_classes=10)
    state = torch.load(args.model, map_location="cpu")
    model.load_state_dict(state)

    # Export
    manifest = export_state_dict(model.state_dict(), args.out_dir, flatten=args.flatten, fmt=args.fmt)
    print(f"Exported {len(manifest['tensors'])} tensors to {args.out_dir}")
    print(f"Manifest: {os.path.join(args.out_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()