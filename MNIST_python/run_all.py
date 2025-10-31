import os
import sys
import argparse
import subprocess


def run(cmd):
    print(f"> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Phase 1 orchestrator: train LeNet-5 on CPU, then evaluate on idx test set")
    # Training args
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--weights_dir", type=str, default="weights")
    parser.add_argument("--weights_path", type=str, default="weights/lenet5.pth")

    # Inference (idx) args
    parser.add_argument("--images", type=str, default="t10k-images.idx3-ubyte", help="Path to MNIST idx test images")
    parser.add_argument("--labels", type=str, default="t10k-labels.idx1-ubyte", help="Path to MNIST idx test labels")
    parser.add_argument("--infer_batch_size", type=int, default=256)
    parser.add_argument("--num_examples", type=int, default=5)

    args = parser.parse_args()

    # Ensure weights directory exists
    os.makedirs(args.weights_dir, exist_ok=True)

    # Train
    train_cmd = (
        f"python train_lenet5.py "
        f"--epochs {args.epochs} "
        f"--batch_size {args.batch_size} "
        f"--lr {args.lr} "
        f"--momentum {args.momentum} "
        f"--num_workers {args.num_workers} "
        f"--weights_dir {args.weights_dir} "
        f"--weights_path {args.weights_path}"
    )
    run(train_cmd)

    # Evaluate on original idx test set
    infer_cmd = (
        f"python infer_idx.py "
        f"--images {args.images} "
        f"--labels {args.labels} "
        f"--model {args.weights_path} "
        f"--batch_size {args.infer_batch_size} "
        f"--num_examples {args.num_examples}"
    )
    run(infer_cmd)

    print("Phase 1 completed: training and idx-based evaluation done.")


if __name__ == "__main__":
    main()