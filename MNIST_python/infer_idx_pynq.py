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
from load_idx import load_mnist_images, load_mnist_labels
import time

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


def program_params(ip, Cin, H, W, Cout, pad, pool, scale_S, shift):
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

def build_stream(in_q_int8, w_q_int8, b_q_int16):
    in_bytes = in_q_int8.reshape(-1).view(np.uint8)
    w_bytes  = w_q_int8.reshape(-1).view(np.uint8)
    b_bytes  = b_q_int16.reshape(-1).view(np.uint8)  # 2 bytes each
    return np.concatenate([in_bytes, w_bytes, b_bytes])

def run_layer_axis_dma(ol, ip_name, dma_name, images, w_int8, b_int16, Cin, H, W, Cout, pad, pool, scale_S, shift):
    ip = getattr(ol, ip_name)
    dma = getattr(ol, dma_name)

    REG_CTRL = write_params_axis_lite(ip, Cin, H, W, Cout, pad, pool, scale_S, shift)

    Cout_o, H_o, W_o = compute_output_shape(Cin, H, W, Cout, pad, pool)
    in_len = Cin * H * W
    out_len = Cout_o * H_o * W_o

    outputs = []
    for img in images:
        payload = build_stream(np.asarray(img, dtype=np.int8),
                               np.asarray(w_int8, dtype=np.int8),
                               np.asarray(b_int16, dtype=np.int16))
        in_buf  = allocate(shape=(payload.size,), dtype=np.uint8)
        out_buf = allocate(shape=(out_len,), dtype=np.uint8)
        np.copyto(in_buf, payload)

        ip.write(REG_CTRL, 1)  # ap_start

        dma.sendchannel.transfer(in_buf)
        dma.recvchannel.transfer(out_buf)
        dma.sendchannel.wait()
        dma.recvchannel.wait()

        out = out_buf.view(np.int8).copy().reshape((Cout_o, H_o, W_o))
        outputs.append(out)

        in_buf.freebuffer()
        out_buf.freebuffer()

    return outputs


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
               image,
               scale_S: float, shift: int,
               conv1_W, conv1_b,
               conv2_W, conv2_b,
               fc1_W, fc1_b,
               fc2_W, fc2_b,
               fc3_W, fc3_b):
    # Stage 1: Conv1+ReLU+Pool on PL
    Cin1, H1, W1 = 1, 28, 28
    fmap1 = run_layer_axis_dma(
        ol, dma_name, ip_name,
        in_map_int8=image.astype(np.int8),
        w_int8=conv1_W,
        b_int8=conv1_b,
        Cin=Cin1, H=H1, W=W1, Cout=6, pad=2, pool=POOL_MAX2x2,
        scale_S=scale_S, shift=shift
    )  # shape: [6, 14, 14] for default

    # Stage 2: Conv2+ReLU+Pool on PL
    Cin2, H2, W2 = 6, 14, 14
    fmap2 = run_layer_axis_dma(
        ol, dma_name, ip_name,
        in_map_int8=fmap1.astype(np.int8),
        w_int8=conv2_W,
        b_int8=conv2_b,
        Cin=Cin2, H=H2, W=W2, Cout=16, pad=0, pool=POOL_MAX2x2,
        scale_S=scale_S, shift=shift
    )  # shape: [16, 5, 5] for default

    # Flatten for FCs (PS)
    flat = fmap2.astype(np.float32).reshape(-1)  # 16*5*5 = 400

    # FC1: 400 -> 120 (float32 on PS)
    h1 = fc1_W @ flat + fc1_b
    relu_inplace(h1)

    # FC2: 120 -> 84
    h2 = fc2_W @ h1 + fc2_b
    relu_inplace(h2)

    # FC3: 84 -> 10
    logits = fc3_W @ h2 + fc3_b

    return logits

# Evaluate saved LeNet-5 model on raw idx test files (CPU)
def evaluate_idx(images_path, labels_path, weights_path):

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

    # Load CNN parameters
    conv1_W = load_csv_matrix(f"{weights_path}/conv1_weight.csv", (6, 1, 5, 5))
    conv1_b = load_csv_vector(f"{weights_path}/conv1_bias.csv", 6)
    conv2_W = load_csv_matrix(f"{weights_path}/conv2_weight.csv", (16, 6, 5, 5))
    conv2_b = load_csv_vector(f"{weights_path}/conv2_bias.csv", 16)
    fc1_W = load_csv_matrix(f"{weights_path}/fc1_weight.csv", (120, 400))  # out x in
    fc1_b = load_csv_vector(f"{weights_path}/fc1_bias.csv", 120)
    fc2_W = load_csv_matrix(f"{weights_path}/fc2_weight.csv", (84, 120))
    fc2_b = load_csv_vector(f"{weights_path}/fc2_bias.csv", 84)
    fc3_W = load_csv_matrix(f"{weights_path}/fc3_weight.csv"ath, (10, 84))
    fc3_b = load_csv_vector(f"{weights_path}/fc3_bias.csv", 10)

    # Run inference
    total = images.shape[0]
    correct = 0
    t0 = time.perf_counter()

    for i in range(total):
        image = images[i]
        label = labels[i]
        # logits = run_lenet5(ol: Overlay, dma_name: str, ip_name: str,
            #    image,
            #    scale_S: float, shift: int,
            #    conv1_W, conv1_b,
            #    conv2_W, conv2_b,
            #    fc1_W, fc1_b,
            #    fc2_W, fc2_b,
            #    fc3_W, fc3_b)
        # probs = softmax(logits)
        # pred = int(np.argmax(probs))
        # pred = argmax(logits)
        # correct += (pred == label)
            
    

    t1 = time.perf_counter()
    total_time_ms = t1 - t0
    accuracy = 100.0 * correct / total
    avg_time_ms = total_time_ms / total
    throughput = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0.0

    print("\n=== Results ===")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"Average inference time: {avg_time_ms:.3f} ms")
    print(f"Throughput: {throughput:.1f} FPS")

def main():
    parser = argparse.ArgumentParser(description="LeNet-5 on Kria: Conv stages on PL via DMA, FCs on PS")
    parser.add_argument("--bitfile", type=str, required=True, help="Path to Overlay (.bit/.hwh)")
    parser.add_argument("--dma_name", type=str, default="axi_dma_0", help="DMA instance name in overlay")
    parser.add_argument("--ip_name", type=str, default="cnn_accel_0", help="Accelerator IP name in overlay")
    parser.add_argument("--images", type=str, default="t10k-images.idx3-ubyte", help="Path to t10k-images.idx3-ubyte")
    parser.add_argument("--labels", type=str, default="t10k-labels.idx1-ubyte", help="Path to t10k-labels.idx1-ubyte")
    parser.add_argument("--weights", type=str, default="weights_csv")
 
    parser.add_argument("--invert", action="store_true", help="Invert image colors before normalization")

    # Conv stage params (defaults align to LeNet-5 modern variant)
    parser.add_argument("--conv1_pad", type=int, default=2)
    parser.add_argument("--conv1_cout", type=int, default=6)
    parser.add_argument("--pool1", type=int, choices=[POOL_NONE, POOL_MAX2x2], default=POOL_MAX2x2)
    parser.add_argument("--conv1_w", type=str, default="weights_csv/conv1_weight.csv")
    parser.add_argument("--conv1_b", type=str, default="weights_csv/conv1_bias.csv")

    parser.add_argument("--conv2_pad", type=int, default=0)
    parser.add_argument("--conv2_cout", type=int, default=16)
    parser.add_argument("--pool2", type=int, choices=[POOL_NONE, POOL_MAX2x2], default=POOL_MAX2x2)
    parser.add_argument("--conv2_w", type=str, default="weights_csv/conv2_weight.csv")
    parser.add_argument("--conv2_b", type=str, default="weights_csv/conv2_bias.csv")

    # Quantization inside the conv IP
    parser.add_argument("--scale_S", type=float, default=1.0, help="Requantization scale in conv IP")
    parser.add_argument("--shift", type=int, default=0, help="Requantization right-shift in conv IP")

    # FC CSVs
    parser.add_argument("--fc1_w", type=str, default="weights_csv/fc1_weight.csv")
    parser.add_argument("--fc1_b", type=str, default="weights_csv/fc1_bias.csv")
    parser.add_argument("--fc2_w", type=str, default="weights_csv/fc2_weight.csv")
    parser.add_argument("--fc2_b", type=str, default="weights_csv/fc2_bias.csv")
    parser.add_argument("--fc3_w", type=str, default="weights_csv/fc3_weight.csv")
    parser.add_argument("--fc3_b", type=str, default="weights_csv/fc3_bias.csv")

    args = parser.parse_args()

    # Load overlay
    ol = load_overlay(args.bitfile)

    evaluate_idx(args.images, args.labels, args.weights)




if __name__ == "__main__":
    main()