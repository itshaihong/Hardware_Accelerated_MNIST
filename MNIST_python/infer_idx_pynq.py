# lenet5_kria_pynq_int8.py
import argparse
import numpy as np
from pynq import Overlay, allocate
from load_idx import load_mnist_images, load_mnist_labels
import time

try:
    from PIL import Image
except Exception:
    Image = None

MEAN = 0.1307
STD  = 0.3081

# ----- Helpers: quantize, pack/unpack -----
def quantize_int8(x_float: np.ndarray, scale: float) -> np.ndarray:
    q = np.rint(x_float * scale).astype(np.int32)
    return np.clip(q, -128, 127).astype(np.int8)

def dequantize_int8(x_int8: np.ndarray, scale: float) -> np.ndarray:
    return x_int8.astype(np.float32) / float(scale)

def pack_int8_to_u32(int8_arr: np.ndarray) -> np.ndarray:
    assert int8_arr.dtype == np.int8
    n = int8_arr.size
    pad = (-n) % 4
    if pad:
        int8_arr = np.concatenate([int8_arr, np.zeros(pad, dtype=np.int8)])
    return int8_arr.view(np.uint8).view(np.uint32)

def pack_int32_to_u32(int32_arr: np.ndarray) -> np.ndarray:
    assert int32_arr.dtype == np.int32
    return int32_arr.view(np.uint32)

def unpack_u32_to_int8(u32_arr: np.ndarray, total_elems: int) -> np.ndarray:
    assert u32_arr.dtype == np.uint32
    return u32_arr.view(np.uint8)[:total_elems].view(np.int8)

# ----- Beat counts (32-bit words) -----
ACT1_ELEMS = 784       # conv1 input image (1x28x28)
W1_ELEMS   = 150       # conv1 weights (6x1x5x5)
B1_ELEMS   = 6
OUT1_ELEMS = 6*12*12   # conv1 pooled output

ACT2_ELEMS = 6*12*12   # conv2 input activations
W2_ELEMS   = 16*6*5*5  # conv2 weights
B2_ELEMS   = 16
OUT2_ELEMS = 16*4*4    # final pooled output

ACT1_BEATS = (ACT1_ELEMS + 3)//4
W1_BEATS   = (W1_ELEMS   + 3)//4
B1_BEATS   = B1_ELEMS
OUT1_BEATS = (OUT1_ELEMS + 3)//4

ACT2_BEATS = (ACT2_ELEMS + 3)//4
W2_BEATS   = (W2_ELEMS   + 3)//4
B2_BEATS   = B2_ELEMS
OUT2_BEATS = (OUT2_ELEMS + 3)//4

# ----- Quantization scales (PLACEHOLDERS — tune to your calibration) -----
# S_out1 is the output scale of conv1 (used to dequantize conv1 outputs or as S_in2).
# S_out2 is the output scale of conv2 (used before FCs).
S_in1   = 32.0
S_w1_pc = np.ones(6, dtype=np.float32) * 64.0    # per-channel conv1 weight scale
S_out1  = 32.0

S_in2   = S_out1
S_w2_pc = np.ones(16, dtype=np.float32) * 64.0   # per-channel conv2 weight scale
S_out2  = 32.0

# Fixed-point requantization: req_m[co]/2^req_shift ≈ (S_in * S_w[co]) / S_out
REQ_SHIFT = 8
def make_req_m(S_in: float, S_w_pc: np.ndarray, S_out: float, req_shift: int) -> np.ndarray:
    alpha = (S_in * S_w_pc) / S_out
    return np.rint(alpha * (1 << req_shift)).astype(np.int32)

REQ_M1 = make_req_m(S_in1, S_w1_pc, S_out1, REQ_SHIFT)  # length 6
REQ_M2 = make_req_m(S_in2, S_w2_pc, S_out2, REQ_SHIFT)  # length 16

# ----- AXI-Lite writers (robust for arrays) -----
def write_axilite_scalars(ip, req_shift_val: int, load_params_val: int):
    rm = getattr(ip, "register_map", None)
    if rm is not None and hasattr(rm, "req_shift"):
        ip.register_map.req_shift = int(req_shift_val)
        ip.register_map.load_params = int(load_params_val)
    else:
        # Fallback: if fields are not exposed, you need to know offsets; adjust accordingly
        # ip.mmio.write(offset_req_shift, req_shift_val)
        # ip.mmio.write(offset_load_params, load_params_val)
        raise RuntimeError("AXI-Lite scalar register offsets not known; expose via register_map or set manually.")

def write_axilite_array(ip, base_offset: int, values: np.ndarray):
    # When req_m is not exposed in register_map, write contiguous 32-bit words from base_offset
    for i, v in enumerate(values.astype(np.uint32)):
        ip.mmio.write(base_offset + 4*i, int(v))

# ----- Parameter load via params DMA -----
def load_params_to_ip(ip, dma_params, w_q_int8: np.ndarray, b_q_int32: np.ndarray):
    payload = np.concatenate([pack_int8_to_u32(w_q_int8), pack_int32_to_u32(b_q_int32)]).astype(np.uint32)
    buf = allocate(shape=(payload.size,), dtype=np.uint32)
    np.copyto(buf, payload)
    # Tell IP to accept params
    ip.register_map.load_params = 1

    if hasattr(ip.register_map, "ap_start"):
        ip.register_map.ap_start = 1
    if hasattr(ip.register_map, "CTRL") and hasattr(ip.register_map.CTRL, "AP_START"):
        ip.register_map.CTRL.AP_START = 1

    buf.flush()
    dma_params.sendchannel.transfer(buf)
    dma_params.sendchannel.wait()
    # Done loading
    ip.register_map.load_params = 0

# ----- Main evaluation using int8 hardware -----
def evaluate_idx_int8(images_path, labels_path, weights_path, overlay,
                      dma_act_name: str, dma_c1_params_name: str, dma_c2_params_name: str,
                      ip_c1_name: str, ip_c2_name: str):

    # Load test set
    images = load_mnist_images(images_path).astype(np.float32)  # [N,1,28,28] in [0,1]
    labels = load_mnist_labels(labels_path).copy()
    if images.shape[0] != labels.shape[0]:
        raise ValueError(f"Number of images ({images.shape[0]}) != number of labels ({labels.shape[0]})")
    images = (images - MEAN) / STD

    # Load float weights/bias (CSV)
    conv1_W_f = np.loadtxt(f"{weights_path}/conv1_weight.csv", delimiter=',', dtype=np.float32).reshape(6, 1, 5, 5)
    conv1_b_f = np.loadtxt(f"{weights_path}/conv1_bias.csv",   delimiter=',', dtype=np.float32).reshape(6)
    conv2_W_f = np.loadtxt(f"{weights_path}/conv2_weight.csv", delimiter=',', dtype=np.float32).reshape(16, 6, 5, 5)
    conv2_b_f = np.loadtxt(f"{weights_path}/conv2_bias.csv",   delimiter=',', dtype=np.float32).reshape(16)

    # Quantize weights and biases
    # Per-channel weight scales
    w1_q = np.clip(np.rint(conv1_W_f * S_w1_pc[:, None, None, None]), -128, 127).astype(np.int8).reshape(-1)
    b1_q = np.rint(conv1_b_f * (S_in1 * S_w1_pc)).astype(np.int32)  # bias in accumulator scale, shape (6,)

    w2_q = np.clip(np.rint(conv2_W_f * S_w2_pc[:, None, None, None]), -128, 127).astype(np.int8).reshape(-1)
    b2_q = np.rint(conv2_b_f * (S_in2 * S_w2_pc)).astype(np.int32)  # shape (16,)

    # Get IPs and DMAs
    ip1 = getattr(overlay, ip_c1_name)
    ip2 = getattr(overlay, ip_c2_name)
    dma_act   = getattr(overlay, dma_act_name)        # MM2S to conv1.act_s, S2MM from conv2.out_s
    dma_c1par = getattr(overlay, dma_c1_params_name)  # MM2S to conv1.param_s
    dma_c2par = getattr(overlay, dma_c2_params_name)  # MM2S to conv2.param_s

    # Write AXI-Lite quantization controls
    # req_shift (scalar) and req_m arrays per IP
    ip1.register_map.req_shift = int(REQ_SHIFT)
    ip2.register_map.req_shift = int(REQ_SHIFT)
    # Write req_m arrays; if register_map exposes them as contiguous memory, you can write via mmio
    # Here we assume Vitis HLS mapped req_m as an array block immediately after req_shift.
    # Replace the offsets below with actual ones if needed.
    # Example offsets (adjust!): offset_req_m1, offset_req_m2
    try:
        # If your register_map shows req_m with a base address:
        base1 = ip1.register_map.Memory_req_m.address
        base2 = ip2.register_map.Memory_req_m.address
        write_axilite_array(ip1, base1, REQ_M1)  # 6 entries
        write_axilite_array(ip2, base2, REQ_M2)  # 16 entries
    except Exception:
        # Fallback: try to write directly if register_map not detailed; adjust offsets to your IP
        # ip1.mmio.write(offset_req_m1 + 4*i, value), etc.
        raise RuntimeError("Please provide AXI-Lite offsets for req_m arrays or ensure register_map exposes them.")

    # Load parameters once via params DMAs
    print("loading 1st layer parameter")
    load_params_to_ip(ip1, dma_c1par, w1_q, b1_q)
    print("loading 2nd layer parameter")
    load_params_to_ip(ip2, dma_c2par, w2_q, b2_q)

    total   = images.shape[0]
    correct = 0
    t0 = time.perf_counter()

    # Allocate activation/output buffers for the activation DMA (uint32 beats)
    in_act_buf  = allocate(shape=(ACT1_BEATS,), dtype=np.uint32)  # image beats
    out_act_buf = allocate(shape=(OUT2_BEATS,), dtype=np.uint32)  # final output beats

    for i in range(total):
        img_f = images[i]        # [1,28,28]
        label = int(labels[i])

        # Quantize image to int8 with S_in1 and pack
        img_q = quantize_int8(img_f.reshape(-1), S_in1)               # 784 int8
        img_beats = pack_int8_to_u32(img_q).astype(np.uint32)         # 196 beats

        # Send activations MM2S → conv1.act_s; receive final outputs S2MM ← conv2.out_s
        np.copyto(in_act_buf, img_beats)
        dma_act.sendchannel.transfer(in_act_buf)
        dma_act.recvchannel.transfer(out_act_buf)
        dma_act.sendchannel.wait()
        dma_act.recvchannel.wait()

        # Unpack final outputs (256 int8) and dequantize for FCs
        out2_q = unpack_u32_to_int8(out_act_buf, total_elems=OUT2_ELEMS)  # (256,) int8
        flat   = dequantize_int8(out2_q, S_out2)                           # (256,) float32

        # FC layers on PS (float32)
        # Load FC weights (assumed shapes: fc1 (120,256), fc2 (84,120), fc3 (10,84))
        # Consider loading these once outside the loop
        fc1_W = np.loadtxt(f"{weights_path}/fc1_weight.csv", delimiter=',', dtype=np.float32).reshape(120, 256)
        fc1_b = np.loadtxt(f"{weights_path}/fc1_bias.csv",   delimiter=',', dtype=np.float32).reshape(120)
        fc2_W = np.loadtxt(f"{weights_path}/fc2_weight.csv", delimiter=',', dtype=np.float32).reshape(84, 120)
        fc2_b = np.loadtxt(f"{weights_path}/fc2_bias.csv",   delimiter=',', dtype=np.float32).reshape(84)
        fc3_W = np.loadtxt(f"{weights_path}/fc3_weight.csv", delimiter=',', dtype=np.float32).reshape(10, 84)
        fc3_b = np.loadtxt(f"{weights_path}/fc3_bias.csv",   delimiter=',', dtype=np.float32).reshape(10)

        h1 = fc1_W @ flat + fc1_b
        h1 = np.maximum(h1, 0.0)

        h2 = fc2_W @ h1 + fc2_b
        h2 = np.maximum(h2, 0.0)

        logits = fc3_W @ h2 + fc3_b
        pred   = int(np.argmax(logits))
        correct += (pred == label)

    t1 = time.perf_counter()
    total_time_s = t1 - t0
    accuracy     = 100.0 * correct / total
    avg_time_ms  = (total_time_s / total) * 1000.0
    throughput   = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0.0

    print("\n=== Results ===")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"Average inference time: {avg_time_ms:.3f} ms")
    print(f"Throughput: {throughput:.1f} FPS")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="LeNet-5 on Kria: Conv stages on PL via DMA (int8), FCs on PS")
    parser.add_argument("--bitfile", type=str, default="MNIST_1.bit", help="Path to Overlay (.bit/.hwh)")
    # Names must match your BD: one activation DMA, two params DMAs
    parser.add_argument("--dma_act_name",      type=str, default="axi_dma_0", help="Activation DMA (MM2S to conv1.act_s, S2MM from conv2.out_s)")
    parser.add_argument("--dma_c1_params_name",type=str, default="axi_dma_1", help="Params DMA for conv1.param_s (MM2S)")
    parser.add_argument("--dma_c2_params_name",type=str, default="axi_dma_2", help="Params DMA for conv2.param_s (MM2S)")
    parser.add_argument("--ip_c1_name",        type=str, default="conv1_preload_axis_0", help="Conv1 IP instance name")
    parser.add_argument("--ip_c2_name",        type=str, default="conv2_preload_axis_0", help="Conv2 IP instance name")
    parser.add_argument("--images", type=str, default="t10k-images.idx3-ubyte")
    parser.add_argument("--labels", type=str, default="t10k-labels.idx1-ubyte")
    parser.add_argument("--weights", type=str, default="weights_csv")
    args = parser.parse_args()

    overlay = Overlay(args.bitfile)
    overlay.download()

    evaluate_idx_int8(args.images, args.labels, args.weights, overlay,
                      args.dma_act_name, args.dma_c1_params_name, args.dma_c2_params_name,
                      args.ip_c1_name, args.ip_c2_name)

