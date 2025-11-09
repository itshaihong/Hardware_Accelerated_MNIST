#!/usr/bin/env python3
import argparse
import json
import math
import numpy as np
import os

def load_csv(path, delimiter=','):
    arr = np.loadtxt(path, delimiter=delimiter)
    if arr.ndim == 0:
        arr = np.array([arr])
    return arr.astype(np.float64)

def symmetric_int8_scale(min_v, max_v, power_of_two=False):
    max_abs = max(abs(min_v), abs(max_v))
    print("max_abs: ", max_abs)
    if max_abs == 0:
        return 1.0, 0  # arbitrary, avoid div0; all weights are zero
    S = 127.0 / max_abs
    if power_of_two:
        n = math.floor(math.log2(S))
        S = float(2 ** n)
    return S, 0  # zero-point is 0

def asymmetric_uint8_scale(min_v, max_v):
    if max_v == min_v:
        # Degenerate: all weights equal. Pick S=1, z to map that value to midrange.
        S = 1.0
        z = int(np.clip(round(-min_v * S), 0, 255))
        return S, z
    S = 255.0 / (max_v - min_v)
    z = int(np.clip(round(-min_v * S), 0, 255))
    return S, z

def quantize(w, S, z, dtype, round_mode='nearest'):
    if dtype == 'int8':
        # symmetric: z should be 0
        if round_mode == 'nearest':
            q = np.round(w * S)
        else:
            q = np.floor(w * S + 0.5)
        q = np.clip(q, -128, 127).astype(np.int8)
        return q
    elif dtype == 'uint8':
        if round_mode == 'nearest':
            q = np.round(w * S + z)
        else:
            q = np.floor(w * S + z + 0.5)
        q = np.clip(q, 0, 255).astype(np.uint8)
        return q
    else:
        raise ValueError("dtype must be 'int8' or 'uint8'")

def per_axis_minmax(arr, axis):
    min_v = np.min(arr, axis=axis, keepdims=False)
    max_v = np.max(arr, axis=axis, keepdims=False)
    return min_v, max_v

def compute_scales(arr, mode, per_axis=None, power_of_two=False):
    # mode: 'symmetric_int8' or 'asymmetric_uint8'
    if per_axis not in (None, 0, 1):
        raise ValueError("per_axis must be None, 0 (per-row), or 1 (per-column)")
    if per_axis is None:
        min_v = float(np.min(arr))
        max_v = float(np.max(arr))
        if mode == 'symmetric_int8':
            S, z = symmetric_int8_scale(min_v, max_v, power_of_two=power_of_two)
        else:
            S, z = asymmetric_uint8_scale(min_v, max_v)
        return np.array([S]), np.array([z]), None
    else:
        # Per-axis scales along selected axis
        min_v, max_v = per_axis_minmax(arr, axis=per_axis)
        S_list = []
        z_list = []
        for mn, mx in zip(np.ravel(min_v), np.ravel(max_v)):
            if mode == 'symmetric_int8':
                S, z = symmetric_int8_scale(float(mn), float(mx), power_of_two=power_of_two)
            else:
                S, z = asymmetric_uint8_scale(float(mn), float(mx))
            S_list.append(S)
            z_list.append(z)
        return np.array(S_list), np.array(z_list), per_axis

def apply_per_axis_quant(arr, S_arr, z_arr, per_axis, dtype):
    # Broadcast along per_axis
    if per_axis is None:
        q = quantize(arr, float(S_arr[0]), int(z_arr[0]), dtype)
    else:
        if per_axis == 0:
            # Per-row: S_arr shape [rows]
            S = S_arr.reshape((-1, 1))
            z = z_arr.reshape((-1, 1))
        else:
            # Per-column: S_arr shape [cols]
            S = S_arr.reshape((1, -1))
            z = z_arr.reshape((1, -1))
        q = quantize(arr, S, z, dtype)
    return q

def dequantize(q, S_arr, z_arr, per_axis, dtype):
    # Return float approximation back from quantized values
    if per_axis is None:
        S = float(S_arr[0])
        z = float(z_arr[0])
        if dtype == 'int8':
            return q.astype(np.float64) / S
        else:
            return (q.astype(np.float64) - z) / S
    else:
        if per_axis == 0:
            S = S_arr.reshape((-1, 1)).astype(np.float64)
            z = z_arr.reshape((-1, 1)).astype(np.float64)
        else:
            S = S_arr.reshape((1, -1)).astype(np.float64)
            z = z_arr.reshape((1, -1)).astype(np.float64)
        if dtype == 'int8':
            return q.astype(np.float64) / S
        else:
            return (q.astype(np.float64) - z) / S

def main():
    ap = argparse.ArgumentParser(description="Compute quantization scales and quantize CSV weights.")
    ap.add_argument("--csv", default="weights_csv/conv2_weight.csv", help="Path to CSV file containing weights")
    ap.add_argument("--delimiter", default=",", help="CSV delimiter (default ,)")
    ap.add_argument("--mode", choices=["symmetric_int8", "asymmetric_uint8"], default="symmetric_int8",
                    help="Quantization mode")
    ap.add_argument("--per-axis", choices=["none", "rows", "cols"], default="none",
                    help="Compute per-tensor (none), per-row (rows), or per-column (cols) scales")
    ap.add_argument("--power-of-two", action="store_true",
                    help="For symmetric_int8, constrain scale to power-of-two (shift-only)")
    ap.add_argument("--save-quantized", default="weights_csv/conv2_weight_quant.csv", help="Path to save quantized weights CSV")
    ap.add_argument("--save-meta", default="weights_csv/conv2_weight_meta.csv", help="Path to save scale/zero-point metadata JSON")
    ap.add_argument("--preview", action="store_true", help="Print summary and error stats")
    args = ap.parse_args()

    arr = load_csv(args.csv, delimiter=args.delimiter)
    per_axis = None if args.per_axis == "none" else (0 if args.per_axis == "rows" else 1)
    dtype = "int8" if args.mode == "symmetric_int8" else "uint8"

    S_arr, z_arr, axis_used = compute_scales(arr, mode=args.mode, per_axis=per_axis, power_of_two=args.power_of_two)
    q = apply_per_axis_quant(arr, S_arr, z_arr, per_axis, dtype)
    dq = dequantize(q, S_arr, z_arr, per_axis, dtype)

    if args.preview:
        min_v, max_v = float(np.min(arr)), float(np.max(arr))
        print(f"Input shape: {arr.shape}")
        print(f"Min: {min_v:.6g}, Max: {max_v:.6g}")
        if per_axis is None:
            print(f"Scale S: {S_arr[0]:.6g}, Zero-point z: {int(z_arr[0])}")
        else:
            print(f"Per-{'row' if per_axis==0 else 'column'} scales computed: {S_arr.shape[0]} items")
        # Error metrics
        mae = float(np.mean(np.abs(arr - dq)))
        rmse = float(np.sqrt(np.mean((arr - dq) ** 2)))
        max_abs_err = float(np.max(np.abs(arr - dq)))
        print(f"MAE: {mae:.6g}, RMSE: {rmse:.6g}, MaxAbsErr: {max_abs_err:.6g}")

    if args.save_quantized:
        np.savetxt(args.save_quantized, q, fmt="%d", delimiter=args.delimiter)
        print(f"Quantized weights saved to {args.save_quantized}")

    min_v, max_v = float(np.min(arr)), float(np.max(arr))
    if args.save_meta:
        meta = {
            "max_abs": max(abs(min_v), abs(max_v)),
            "mode": args.mode,
            "dtype": dtype,
            "shape": arr.shape,
            "per_axis": "none" if per_axis is None else ("rows" if per_axis == 0 else "cols"),
            "power_of_two": args.power_of_two,
            "scale": S_arr.tolist(),
            "zero_point": z_arr.tolist(),
            "notes": (
                "For symmetric_int8, real ≈ q / S. "
                "For asymmetric_uint8, real ≈ (q - z) / S. "
                "If power_of_two is True (symmetric), S=2^n for shift-only requantization."
            )
        }
        with open(args.save_meta, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Metadata saved to {args.save_meta}")

if __name__ == "__main__":
    main()