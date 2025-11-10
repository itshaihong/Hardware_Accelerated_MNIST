# lenet5_kria_pynq_int8.py
from pynq import Overlay, allocate
import time
import sys
import os
import time
import argparse
import numpy as np
from PIL import Image, ImageOps



OVERLAY_PATH = "fc1.bit" 
DMA0_NAME = "axi_dma_0"  
ol = Overlay(OVERLAY_PATH)
dma0 = getattr(ol, DMA0_NAME) 


input_buf_fc1 = allocate(shape=(400 + 400*120,), dtype=np.int32)
output_buf_fc1 = allocate(shape=(120,), dtype=np.int32)



def dma0_transfer(in_buf, out_buf):
    dma0.recvchannel.transfer(out_buf)
    dma0.sendchannel.transfer(in_buf)
    dma0.sendchannel.wait()
    dma0.recvchannel.wait()

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


# ----- Main evaluation using int8 hardware -----
def evaluate_idx_int8(image, label, weights_path):

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

    t0 = time.perf_counter()

    c1 = lenet5_layer1_conv_maxpool(image, conv1_W_f, conv1_b_f)
    c2 = lenet5_layer2_conv_maxpool(c1, conv2_W_f, conv2_b_f)

    flat = c2.reshape(-1) 

    h1 = fc1_W @ flat + fc1_b
    h1 = np.maximum(h1, 0.0)

    h2 = fc2_W @ h1 + fc2_b
    h2 = np.maximum(h2, 0.0)
    logits = fc3_W @ h2 + fc3_b
    pred   = int(np.argmax(logits))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Prediction: {pred}, elapsed: {elapsed_ms:.3f} ms")

    return pred, elapsed_ms



# Open image robustly and composite transparency onto white background if needed
def open_grayscale_robust(image_path):
    img = Image.open(image_path)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg.convert("L")
    elif img.mode == "LA":
        l, a = img.split()
        bg = Image.new("L", img.size, 255)
        bg.paste(l, mask=a)
        img = bg
    elif img.mode == "P":
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg.convert("L")
    else:
        img = img.convert("L")
    return img

# Preprocess: ensure 28x28, optional invert, normalize as training
def load_and_preprocess(image_path, invert=False, auto_invert=True, save_preprocessed=None):
    img = open_grayscale_robust(image_path)

    if img.size != (28, 28):
        img = img.resize((28, 28), resample=Image.BILINEAR)

    arr01 = np.array(img).astype(np.float32) / 255.0

    if auto_invert and not invert:
        mean_intensity = float(arr01.mean())
        if mean_intensity > 0.5:
            img = ImageOps.invert(img)
            arr01 = 1.0 - arr01

    if invert and not auto_invert:
        img = ImageOps.invert(img)
        arr01 = 1.0 - arr01

    if save_preprocessed:
        os.makedirs(os.path.dirname(save_preprocessed) or ".", exist_ok=True)
        img.save(save_preprocessed)

    mean, std = 0.1307, 0.3081
    arr_norm = (arr01 - mean) / std
    np.savetxt('test_3_norm.csv', arr_norm, fmt="%.7g", delimiter=',')

    print(f"Preprocess stats: min={arr01.min():.3f}, max={arr01.max():.3f}, mean={arr01.mean():.3f}")
    return arr_norm[np.newaxis, :, :] 

def predict_single(image_path,
                   invert=False, auto_invert=True,
                   save_preprocessed=None, topk=3):

    x = load_and_preprocess(
        image_path,
        invert=invert,
        auto_invert=auto_invert,
        save_preprocessed=save_preprocessed
    )


    evaluate_idx_int8(x, 3, "weights_csv/")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a single 28x28 MNIST-like grayscale image with LeNet-5 (CPU)")
    parser.add_argument("--image", default="test_3.png", type=str, required=True, help="Path to a 28x28 grayscale PNG/JPG")
    parser.add_argument("--invert", action="store_true", help="Force invert (white background -> black background)")
    parser.add_argument("--no_auto_invert", action="store_true", help="Disable auto-invert by mean intensity")
    parser.add_argument("--save_preprocessed", type=str, default="", help="Optional path to save the preprocessed 28x28 image")
    parser.add_argument("--topk", type=int, default=3, help="Show top-K probabilities")
    args = parser.parse_args()

    predict_single(
        image_path=args.image,
        invert=args.invert,
        auto_invert=(not args.no_auto_invert),
        save_preprocessed=(args.save_preprocessed if args.save_preprocessed else None),
        topk=args.topk
    )

