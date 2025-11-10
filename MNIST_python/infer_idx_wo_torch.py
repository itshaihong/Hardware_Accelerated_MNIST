import os
import time
import argparse
import numpy as np
import torch
from lenet5 import LeNet5
from load_idx import load_mnist_images, load_mnist_labels



def conv2d(x, filters, bias=None, padding=0, stride=1):
    """
    x: numpy array of shape (C_in, H, W)
    filters: numpy array of shape (C_out, C_in, K, K)
    bias: numpy array of shape (C_out,) or None
    padding: integer padding applied equally on all sides
    stride: integer stride (default 1)
    Returns: numpy array of shape (C_out, H_out, W_out)
    """
    C_in, H, W = x.shape
    C_out, C_f_in, K, K2 = filters.shape
    assert C_in == C_f_in, "Filter input channels must match x channels."
    assert K == K2, "Filters must be square."

    # Pad input
    if padding > 0:
        x_padded = np.pad(x, ((0, 0), (padding, padding), (padding, padding)), mode='constant')
    else:
        x_padded = x

    H_p, W_p = x_padded.shape[1], x_padded.shape[2]
    H_out = (H_p - K) // stride + 1
    W_out = (W_p - K) // stride + 1

    out = np.zeros((C_out, H_out, W_out), dtype=x.dtype)

    for oc in range(C_out):
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                region = x_padded[:, h_start:h_start + K, w_start:w_start + K]  # (C_in, K, K)
                out[oc, i, j] = np.sum(region * filters[oc, :, :, :])
        if bias is not None:
            out[oc] += bias[oc]
    return out

def maxpool2x2(x):
    """
    x: numpy array of shape (C, H, W)
    Returns: numpy array of shape (C, H//2, W//2)
    """
    C, H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0, "H and W must be even for 2x2 stride-2 maxpool."
    H_out, W_out = H // 2, W // 2
    out = np.zeros((C, H_out, W_out), dtype=x.dtype)
    for c in range(C):
        for i in range(H_out):
            for j in range(W_out):
                h_start = 2 * i
                w_start = 2 * j
                patch = x[c, h_start:h_start + 2, w_start:w_start + 2]
                out[c, i, j] = np.max(patch)
    return out

def lenet5_layer1_conv_maxpool(x, filters, bias=None):
    """
    Layer 1: input 1x28x28, padding=2, conv 5x5 -> 6x28x28, then 2x2 maxpool -> 6x14x14
    x: numpy array of shape (1, 28, 28)
    filters: numpy array of shape (6, 1, 5, 5)
    bias: numpy array of shape (6,) or None
    Returns: numpy array of shape (6, 14, 14)
    """
    # Convolution with padding=2, stride=1
    conv_out = conv2d(x, filters, bias=bias, padding=2, stride=1)  # (6, 28, 28)
    conv_out = np.maximum(conv_out, 0)
    # Max-pooling 2x2, stride 2
    pooled = maxpool2x2(conv_out)  # (6, 14, 14)
    return pooled

def lenet5_layer2_conv_maxpool(x, filters, bias=None):
    """
    Layer 2: input 6x14x14, padding=0, conv 5x5 -> 16x10x10, then 2x2 maxpool -> 16x5x5
    x: numpy array of shape (6, 14, 14)
    filters: numpy array of shape (16, 6, 5, 5)
    bias: numpy array of shape (16,) or None
    Returns: numpy array of shape (16, 5, 5)
    """
    # Convolution with padding=0, stride=1
    conv_out = conv2d(x, filters, bias=bias, padding=0, stride=1)  # (16, 10, 10)
    conv_out = np.maximum(conv_out, 0)
    # Max-pooling 2x2, stride 2
    pooled = maxpool2x2(conv_out)  # (16, 5, 5)
    return pooled



# Evaluate saved LeNet-5 model on raw idx test files (CPU)
def evaluate_idx(total, images_path, labels_path, weights_path='weights_csv/'):
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

    # Load float weights/bias (CSV)
    conv1_W_f = np.loadtxt(f"{weights_path}/conv1_weight.csv", delimiter=',', dtype=np.float32).reshape(6, 1, 5, 5)
    conv1_b_f = np.loadtxt(f"{weights_path}/conv1_bias.csv",   delimiter=',', dtype=np.float32).reshape(6)

    conv2_W_f = np.loadtxt(f"{weights_path}/conv2_weight.csv", delimiter=',', dtype=np.float32).reshape(16, 6, 5, 5)
    conv2_b_f = np.loadtxt(f"{weights_path}/conv2_bias.csv",   delimiter=',', dtype=np.float32).reshape(16)

    fc1_W = np.loadtxt(f"{weights_path}/fc1_weight.csv", delimiter=',', dtype=np.float32).reshape(120, 400)
    fc1_b = np.loadtxt(f"{weights_path}/fc1_bias.csv",   delimiter=',', dtype=np.float32).reshape(120)
    fc2_W = np.loadtxt(f"{weights_path}/fc2_weight.csv", delimiter=',', dtype=np.float32).reshape(84, 120)
    fc2_b = np.loadtxt(f"{weights_path}/fc2_bias.csv",   delimiter=',', dtype=np.float32).reshape(84)
    fc3_W = np.loadtxt(f"{weights_path}/fc3_weight.csv", delimiter=',', dtype=np.float32).reshape(10, 84)
    fc3_b = np.loadtxt(f"{weights_path}/fc3_bias.csv",   delimiter=',', dtype=np.float32).reshape(10)


    # Run inference
    correct = 0
    total_time_ms = 0.0

    # Print stats for first K samples
    print(f"Loaded {total} test images from idx files")
    for i in range(total):

        t0 = time.perf_counter()

        c1 = lenet5_layer1_conv_maxpool(images[i], conv1_W_f, conv1_b_f)
        c2 = lenet5_layer2_conv_maxpool(c1, conv2_W_f, conv2_b_f)
        flat = c2.reshape(-1) 

        h1 = fc1_W @ flat + fc1_b
        h1 = np.maximum(h1, 0.0)

        h2 = fc2_W @ h1 + fc2_b
        h2 = np.maximum(h2, 0.0)
        logits = fc3_W @ h2 + fc3_b
        t1 = time.perf_counter()

        pred  = int(np.argmax(logits))
        total_time_ms += (t1 - t0) * 1000.0

        if (i < 10):
            print(f"Image {i}: Prediction: {pred}, True Label: {labels[i]}")
        if pred == labels[i]:
            correct = correct + 1

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
    args = parser.parse_args()

    evaluate_idx(100, args.images, args.labels)
