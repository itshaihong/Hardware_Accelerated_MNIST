import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F

from lenet5 import LeNet5
from load_idx import load_idx_test

# Training-time normalization constants
MEAN = 0.1307
STD = 0.3081


def preprocess_idx_images(images_np: np.ndarray) -> torch.Tensor:
    """
    images_np: (N, 28, 28) float32 in [0,1]
    Returns a torch tensor normalized with training params: (N, 1, 28, 28)
    """
    # Convert to tensor and add channel dimension
    x = torch.from_numpy(images_np).unsqueeze(1)  # (N,1,28,28)
    # Apply Normalize((MEAN,), (STD,)) equivalent: (x - mean) / std
    x = (x - MEAN) / STD
    return x


def batch_infer(model, x: torch.Tensor, labels_np: np.ndarray, batch_size: int = 256):
    """
    Run batch inference and compute stats.
    Timing includes only the forward pass, excluding preprocessing and concatenation.
    """
    model.eval()
    device = torch.device("cpu")  # Phase 1 CPU-only
    correct = 0
    total = x.shape[0]

    # Measure forward pass total time
    forward_time = 0.0

    with torch.no_grad():
        for i in range(0, total, batch_size):
            xb = x[i:i + batch_size].to(device)

            start = time.perf_counter()
            logits = model(xb)
            end = time.perf_counter()
            forward_time += (end - start)

            preds = logits.argmax(dim=1).cpu().numpy()
            correct += (preds == labels_np[i:i + batch_size]).sum()

    accuracy = correct / total
    avg_latency_ms = (forward_time / total) * 1000.0
    fps = total / forward_time if forward_time > 0 else float("inf")
    return accuracy, avg_latency_ms, fps


def print_examples(model, x: torch.Tensor, labels_np: np.ndarray, num_examples: int = 5):
    """
    Print first few examples: label, prediction, top-3 probabilities.
    """
    model.eval()
    with torch.no_grad():
        xb = x[:num_examples]
        logits = model(xb)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    for i in range(num_examples):
        top3_idx = probs[i].argsort()[-3:][::-1]
        top3 = [(int(k), float(probs[i][k])) for k in top3_idx]
        print(f"Example {i}: Label={int(labels_np[i])}, Pred={int(preds[i])}, Top-3={top3}")


def main():
    parser = argparse.ArgumentParser(description="Batch inference on original MNIST idx test set (CPU)")
    parser.add_argument("--images", type=str, default="t10k-images.idx3-ubyte", help="Path to MNIST idx test images")
    parser.add_argument("--labels", type=str, default="t10k-labels.idx1-ubyte", help="Path to MNIST idx test labels")
    parser.add_argument("--model", type=str, default="weights/lenet5.pth", help="Path to model weights")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_examples", type=int, default=5)
    args = parser.parse_args()

    # Load data
    images_np, labels_np = load_idx_test(args.images, args.labels)

    # Preprocess
    x = preprocess_idx_images(images_np)  # (N,1,28,28)

    # Load model
    model = LeNet5(num_classes=10)
    state = torch.load(args.model, map_location="cpu")
    model.load_state_dict(state)

    # Inference stats
    accuracy, avg_latency_ms, fps = batch_infer(model, x, labels_np, batch_size=args.batch_size)
    print(f"Accuracy (idx test set): {accuracy:.4f}")
    print(f"Average inference latency: {avg_latency_ms:.3f} ms/sample")
    print(f"Throughput: {fps:.2f} FPS")

    # Print a few examples
    print_examples(model, x, labels_np, num_examples=args.num_examples)


if __name__ == "__main__":
    main()