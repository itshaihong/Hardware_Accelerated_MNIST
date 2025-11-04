import struct
import numpy as np

# Load MNIST idx image file (uint8), return float32 array normalized to [0,1]
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError("Invalid magic number for images file")
        images = np.frombuffer(f.read(num * rows * cols), dtype=np.uint8)
        images = images.reshape(num, 1, rows, cols).astype(np.float32) / 255.0
        return images

# Load MNIST idx label file (uint8), return numpy array of labels
def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError("Invalid magic number for labels file")
        labels = np.frombuffer(f.read(num), dtype=np.uint8)
        return labels