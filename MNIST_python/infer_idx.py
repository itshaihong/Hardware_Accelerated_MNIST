import os
import time
import argparse
import numpy as np
import torch
from lenet5 import LeNet5
from load_idx import load_mnist_images, load_mnist_labels

# Evaluate saved LeNet-5 model on raw idx test files (CPU)
def evaluate_idx(images_path, labels_path, model_path='weights/lenet5.pth', batch_size=1000, print_samples=10):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load idx data
    images = load_mnist_images(images_path)  # shape [N,1,28,28] in [0,1]
    labels = load_mnist_labels(labels_path)  # shape [N]
    labels = labels.copy()
    if images.shape[0] != labels.shape[0]:
        raise ValueError(f"Number of images ({images.shape[0]}) != number of labels ({labels.shape[0]})")

    # Normalize using the same mean/std as training
    mean = 0.1307
    std = 0.3081
    images = (images - mean) / std

    # Convert to torch tensors
    device = torch.device('cpu')
    X = torch.from_numpy(images).float()
    y = torch.from_numpy(labels).long()

    # Load model
    model = LeNet5().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Run inference
    total = X.shape[0]
    correct = 0
    total_time_ms = 0.0

    # Print stats for first K samples
    print(f"Loaded {total} test images from idx files")
    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_X = X[start:end].to(device)
            batch_y = y[start:end].to(device)

            t0 = time.perf_counter()
            logits = model(batch_X)
            t1 = time.perf_counter()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total_time_ms += (t1 - t0) * 1000.0

            # Print first few predictions
            if start == 0:
                k = min(print_samples, batch_X.shape[0])
                for i in range(k):
                    print(f"Image {i}: Prediction: {int(preds[i].item())}, True Label: {int(batch_y[i].item())}")

    accuracy = 100.0 * correct / total
    avg_time_ms = total_time_ms / total
    throughput = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0.0

    print("\n=== Results ===")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"Average inference time: {avg_time_ms:.3f} ms")
    print(f"Throughput: {throughput:.1f} FPS")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LeNet-5 on raw idx MNIST test files (CPU)")
    parser.add_argument("--images", type=str, default="t10k-images.idx3-ubyte", help="Path to t10k-images.idx3-ubyte")
    parser.add_argument("--labels", type=str, default="t10k-labels.idx1-ubyte", help="Path to t10k-labels.idx1-ubyte")
    parser.add_argument("--model", type=str, default="weights/lenet5.pth", help="Path to saved model .pth")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for inference")
    args = parser.parse_args()

    evaluate_idx(args.images, args.labels, model_path=args.model, batch_size=args.batch_size)