#!/usr/bin/env python3
# lenet5_kria_pynq.py
#
# LeNet-5 inference on Kria:
# - Stage 1 (Conv1+ReLU+Pool) on PL via AXI-Stream DMA
# - Stage 2 (Conv2+ReLU+Pool) on PL via AXI-Stream DMA
# - FC1, FC2, FC3 on PS (NumPy)
#
# Assumptions:
# - Overlay (.bit/.hwh) includes:
#   * AXI DMA instance (MM2S/S2MM)
#   * CNN accelerator IP (AXI-Lite control + AXI-Stream in/out)
#   * Conv IP applies Conv5x5 + ReLU + optional MaxPool2x2 + requant int8
# - Conv IP weights/biases are already loaded/configured (not streamed here).
# - FC weights/biases are provided as CSV files (float32 recommended).
#
# Usage example:
#   python3 lenet5_kria_pynq.py \
#     --bitfile /home/xilinx/overlays/cnn_accel.bit \
#     --dma_name axi_dma_0 \
#     --ip_name cnn_accel_0 \
#     --fc1_w fc1_weight.csv --fc1_b fc1_bias.csv \
#     --fc2_w fc2_weight.csv --fc2_b fc2_bias.csv \
#     --fc3_w fc3_weight.csv --fc3_b fc3_bias.csv \
#     --image digit.png --invert
#
# If you want me to be able to save or delete personalisations, enable the feature on the personalisations page (https://ai-know.nus.edu.sg/personalisation).

import argparse
import numpy as np
from pynq import Overlay, allocate

try:
    from PIL import Image
except Exception:
    Image = None

# Match your HLS defines
POOL_NONE = 0
POOL_MAX2x2 = 1

# Training normalization constants for MNIST
MEAN = 0.1307
STD = 0.3081


def load_overlay(bitfile_path: str) -> Overlay:
    ol = Overlay(bitfile_path)
    ol.download()
    return ol


def preprocess_image_28x28(path: str, invert: bool = False, normalize: bool = True, to_int8: bool = True) -> np.ndarray:
    if Image is None:
        raise RuntimeError("PIL is not available. Install pillow or use --random instead of --image.")
    img = Image.open(path).convert("L")
    if img.size != (28, 28):
        img = img.resize((28, 28), resample=Image.NEAREST)
    x = np.asarray(img, dtype=np.float32) / 255.0
    if invert:
        x = 1.0 - x
    if normalize:
        x = (x - MEAN) / STD
    if to_int8:
        # Simple input quantization to int8; you can calibrate scale
        scale_in = 32.0
        q = np.clip((x * scale_in).astype(np.int32), -128, 127).astype(np.int8)
        return q
    return x


def compute_output_shape(Cin: int, H: int, W: int, Cout: int, pad: int, pool: int):
    Hout_conv = H + 2 * pad - 5 + 1
    Wout_conv = W + 2 * pad - 5 + 1
    if pool == POOL_NONE:
        return Cout, Hout_conv, Wout_conv
    else:
        return Cout, Hout_conv // 2, Wout_conv // 2


def write_params_axis_lite(ip, Cin, H, W, Cout, pad, pool, scale_S, shift):
    # Adjust these offsets to match your IP register map
    REG_CTRL  = 0x00  # bit0 ap_start
    REG_CIN   = 0x10
    REG_H     = 0x14
    REG_W     = 0x18
    REG_COUT  = 0x1C
    REG_PAD   = 0x20
    REG_POOL  = 0x24
    REG_SCALE = 0x28  # write IEEE-754 bits if register expects float
    REG_SHIFT = 0x2C

    ip.write(REG_CIN,   Cin)
    ip.write(REG_H,     H)
    ip.write(REG_W,     W)
    ip.write(REG_COUT,  Cout)
    ip.write(REG_PAD,   pad)
    ip.write(REG_POOL,  pool)
    ip.write(REG_SCALE, np.float32(scale_S).view(np.uint32).item())
    ip.write(REG_SHIFT, shift)

    return REG_CTRL


def run_conv_axis_dma(ol: Overlay, dma_name: str, ip_name: str,
                      in_map_int8: np.ndarray,
                      Cin: int, H: int, W: int, Cout: int, pad: int, pool: int,
                      scale_S: float, shift: int) -> np.ndarray:
    ip = getattr(ol, ip_name)
    dma = getattr(ol, dma_name)

    REG_CTRL = write_params_axis_lite(ip, Cin, H, W, Cout, pad, pool, scale_S, shift)

    Cout_o, H_o, W_o = compute_output_shape(Cin, H, W, Cout, pad, pool)
    in_len = Cin * H * W
    out_len = Cout_o * H_o * W_o

    in_buf = allocate(shape=(in_len,), dtype=np.int8)
    out_buf = allocate(shape=(out_len,), dtype=np.int8)

    if in_map_int8.size != in_len:
        raise ValueError(f"Input size mismatch: expected {in_len}, got {in_map_int8.size}")
    np.copyto(in_buf, in_map_int8.reshape(-1))

    # Start the core
    ip.write(REG_CTRL, 1)  # ap_start

    # Stream input to IP, receive output from IP
    dma.sendchannel.transfer(in_buf)
    dma.recvchannel.transfer(out_buf)
    dma.sendchannel.wait()
    dma.recvchannel.wait()

    out_np = np.array(out_buf, dtype=np.int8).reshape((Cout_o, H_o, W_o))

    in_buf.freebuffer()
    out_buf.freebuffer()

    return out_np


def load_csv_matrix(path: str, shape: tuple) -> np.ndarray:
    """Load a CSV matrix and reshape to given shape. Assumes row-major CSV."""
    mat = np.loadtxt(path, delimiter=',', dtype=np.float32)
    flat = mat.reshape(-1)
    expected = int(np.prod(shape))
    if flat.size != expected:
        raise ValueError(f"{path} has {flat.size} elements, expected {expected} for shape {shape}")
    return flat.reshape(shape)


def load_csv_vector(path: str, length: int) -> np.ndarray:
    vec = np.loadtxt(path, delimiter=',', dtype=np.float32).reshape(-1)
    if vec.size != length:
        raise ValueError(f"{path} has {vec.size} elements, expected {length}")
    return vec


def relu_inplace(x: np.ndarray):
    np.maximum(x, 0.0, out=x)


def softmax(x: np.ndarray) -> np.ndarray:
    m = np.max(x)
    e = np.exp(x - m)
    return e / np.sum(e)


def run_lenet5(ol: Overlay, dma_name: str, ip_name: str,
               img_or_random: np.ndarray,
               conv1_pad: int, conv1_Cout: int, pool1_mode: int,
               conv2_pad: int, conv2_Cout: int, pool2_mode: int,
               scale_S: float, shift: int,
               fc1_w_path: str, fc1_b_path: str,
               fc2_w_path: str, fc2_b_path: str,
               fc3_w_path: str, fc3_b_path: str):
    # Stage 1: Conv1+ReLU+Pool on PL
    Cin1, H1, W1 = 1, 28, 28
    fmap1 = run_conv_axis_dma(
        ol, dma_name, ip_name,
        in_map_int8=img_or_random.reshape(1, H1, W1).astype(np.int8),
        Cin=Cin1, H=H1, W=W1, Cout=conv1_Cout, pad=conv1_pad, pool=pool1_mode,
        scale_S=scale_S, shift=shift
    )  # shape: [6, 14, 14] for default

    # Stage 2: Conv2+ReLU+Pool on PL
    Cin2 = conv1_Cout
    H2, W2 = fmap1.shape[1], fmap1.shape[2]  # 14x14
    fmap2 = run_conv_axis_dma(
        ol, dma_name, ip_name,
        in_map_int8=fmap1.astype(np.int8),
        Cin=Cin2, H=H2, W=W2, Cout=conv2_Cout, pad=conv2_pad, pool=pool2_mode,
        scale_S=scale_S, shift=shift
    )  # shape: [16, 5, 5] for default

    # Flatten for FCs (PS)
    flat = fmap2.astype(np.float32).reshape(-1)  # 16*5*5 = 400

    # FC1: 400 -> 120 (float32 on PS)
    fc1_W = load_csv_matrix(fc1_w_path, (120, 400))  # out x in
    fc1_b = load_csv_vector(fc1_b_path, 120)
    h1 = fc1_W @ flat + fc1_b
    relu_inplace(h1)

    # FC2: 120 -> 84
    fc2_W = load_csv_matrix(fc2_w_path, (84, 120))
    fc2_b = load_csv_vector(fc2_b_path, 84)
    h2 = fc2_W @ h1 + fc2_b
    relu_inplace(h2)

    # FC3: 84 -> 10
    fc3_W = load_csv_matrix(fc3_w_path, (10, 84))
    fc3_b = load_csv_vector(fc3_b_path, 10)
    logits = fc3_W @ h2 + fc3_b

    probs = softmax(logits)
    pred = int(np.argmax(probs))
    return pred, probs, (fmap1, fmap2, h1, h2, logits)


def main():
    parser = argparse.ArgumentParser(description="LeNet-5 on Kria: Conv stages on PL via DMA, FCs on PS")
    parser.add_argument("--bitfile", type=str, required=True, help="Path to Overlay (.bit/.hwh)")
    parser.add_argument("--dma_name", type=str, default="axi_dma_0", help="DMA instance name in overlay")
    parser.add_argument("--ip_name", type=str, default="cnn_accel_0", help="Accelerator IP name in overlay")
    parser.add_argument("--image", type=str, default=None, help="28x28 PNG for inference")
    parser.add_argument("--invert", action="store_true", help="Invert image colors before normalization")
    parser.add_argument("--random", action="store_true", help="Use random input instead of image")

    # Conv stage params (defaults align to LeNet-5 modern variant)
    parser.add_argument("--conv1_pad", type=int, default=2)
    parser.add_argument("--conv1_cout", type=int, default=6)
    parser.add_argument("--pool1", type=int, choices=[POOL_NONE, POOL_MAX2x2], default=POOL_MAX2x2)

    parser.add_argument("--conv2_pad", type=int, default=0)
    parser.add_argument("--conv2_cout", type=int, default=16)
    parser.add_argument("--pool2", type=int, choices=[POOL_NONE, POOL_MAX2x2], default=POOL_MAX2x2)

    # Quantization inside the conv IP
    parser.add_argument("--scale_S", type=float, default=1.0, help="Requantization scale in conv IP")
    parser.add_argument("--shift", type=int, default=0, help="Requantization right-shift in conv IP")

    # FC CSVs
    parser.add_argument("--fc1_w", type=str, required=True, help="Path to fc1_weight.csv (120x400)")
    parser.add_argument("--fc1_b", type=str, required=True, help="Path to fc1_bias.csv (120)")
    parser.add_argument("--fc2_w", type=str, required=True, help="Path to fc2_weight.csv (84x120)")
    parser.add_argument("--fc2_b", type=str, required=True, help="Path to fc2_bias.csv (84)")
    parser.add_argument("--fc3_w", type=str, required=True, help="Path to fc3_weight.csv (10x84)")
    parser.add_argument("--fc3_b", type=str, required=True, help="Path to fc3_bias.csv (10)")

    args = parser.parse_args()

    # Load overlay
    ol = load_overlay(args.bitfile)

    # Prepare input
    if args.random:
        img_int8 = np.random.randint(-128, 128, size=(28, 28), dtype=np.int8)
    elif args.image:
        img_int8 = preprocess_image_28x28(args.image, invert=args.invert, normalize=True, to_int8=True)
    else:
        raise ValueError("Provide --image path or use --random to generate input.")

    # Run end-to-end
    pred, probs, _ = run_lenet5(
        ol, args.dma_name, args.ip_name,
        img_or_random=img_int8,
        conv1_pad=args.conv1_pad, conv1_Cout=args.conv1_cout, pool1_mode=args.pool1,
        conv2_pad=args.conv2_pad, conv2_Cout=args.conv2_cout, pool2_mode=args.pool2,
        scale_S=args.scale_S, shift=args.shift,
        fc1_w_path=args.fc1_w, fc1_b_path=args.fc1_b,
        fc2_w_path=args.fc2_w, fc2_b_path=args.fc2_b,
        fc3_w_path=args.fc3_w, fc3_b_path=args.fc3_b
    )

    # Report
    print(f"Prediction: {pred}")
    print("Top-10 probabilities:")
    for i, p in enumerate(probs):
        print(f"  class {i}: {p:.4f}")


if __name__ == "__main__":
    main()