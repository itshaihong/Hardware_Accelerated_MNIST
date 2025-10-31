import gzip
import struct
import numpy as np
from typing import Tuple


def _open_maybe_gzip(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


def load_mnist_idx_images(path: str) -> np.ndarray:
    """
    Loads MNIST images from idx3-ubyte (optionally .gz).
    Returns float32 array of shape (N, 28, 28) scaled to [0,1].
    """
    with _open_maybe_gzip(path) as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number for images: {magic}")
        buffer = f.read(rows * cols * num)
        data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num, rows, cols)
        data /= 255.0
        return data


def load_mnist_idx_labels(path: str) -> np.ndarray:
    """
    Loads MNIST labels from idx1-ubyte (optionally .gz).
    Returns uint8 array of shape (N,).
    """
    with _open_maybe_gzip(path) as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number for labels: {magic}")
        buffer = f.read(num)
        labels = np.frombuffer(buffer, dtype=np.uint8)
        return labels


def load_idx_test(images_path: str, labels_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to load the MNIST test set from original idx files.

    images: t10k-images.idx3-ubyte or .gz
    labels: t10k-labels.idx1-ubyte or .gz

    Returns:
      images_float: (N, 28, 28) float32 in [0,1]
      labels: (N,) uint8
    """
    images = load_mnist_idx_images(images_path)
    labels = load_mnist_idx_labels(labels_path)

    if images.shape[0] != labels.shape[0]:
        raise ValueError(f"Image count {images.shape[0]} != label count {labels.shape[0]}")

    return images, labels