// tb_conv2_int8_axilite_axis.cpp
// Testbench for conv2_int8_axilite_axis (LeNet-5 Conv2 + ReLU + MaxPool with int8 quantization).
// Payloads:
//  - act_s: int8 activations packed 4 per 32-bit beat
//  - param_s: first W_ELEMS int8 weights packed 4 per beat, then B_ELEMS int32 bias (1 per beat)
//  - out_s: int8 outputs packed 4 per 32-bit beat
//
// AXI-Lite controls:
//  - load_params: 1 to load weights/bias from param_s, 0 to reuse already loaded
//  - req_shift: common right shift for requantization
//  - req_m[C_OUT]: per-channel fixed-point multipliers for requantization

#include <iostream>
#include <vector>
#include <cstdint>
#include <cassert>
#include <cmath>

#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>

typedef ap_axiu<32,0,0,0> axis_t;

// LeNet-5 Conv1 fixed sizes (must match kernel)
constexpr int C_IN   = 1;   // from Pool1
constexpr int H_IN   = 32;
constexpr int W_IN   = 32;
constexpr int C_OUT  = 6;  // Conv2 filters
constexpr int K      = 5;

constexpr int H_CONV = H_IN - K + 1; // 24
constexpr int W_CONV = W_IN - K + 1; // 24

constexpr int S_POOL = 2;            // 2x2 stride 2
constexpr int H_OUT  = H_CONV / S_POOL; // 12
constexpr int W_OUT  = W_CONV / S_POOL; // 12

// Element counts
constexpr int ACT_ELEMS   = C_IN * H_IN    * W_IN;     // 784
constexpr int W_ELEMS     = C_OUT * C_IN * K * K;      // 150
constexpr int B_ELEMS     = C_OUT;                     // 6
constexpr int CONV_ELEMS  = C_OUT * H_CONV * W_CONV;   // 3456
constexpr int OUT_ELEMS   = C_OUT * H_OUT  * W_OUT;    // 864

// Beat counts (32-bit words)
constexpr int ACT_BEATS   = (ACT_ELEMS + 3) / 4;       // 216
constexpr int W_BEATS     = (W_ELEMS   + 3) / 4;       // 600
constexpr int B_BEATS     = B_ELEMS;                   // 16
constexpr int OUT_BEATS   = (OUT_ELEMS  + 3) / 4;      // 64

// Index helpers (match kernel)
static inline int idx_act(int c, int y, int x)          { return (c*H_IN + y)*W_IN + x; }
static inline int idx_w  (int co, int ci, int ky, int kx){ return ((co*C_IN + ci)*K + ky)*K + kx; }
static inline int idx_c  (int co, int y, int x)         { return (co*H_CONV + y)*W_CONV + x; }
static inline int idx_out(int co, int y, int x)         { return (co*H_OUT  + y)*W_OUT  + x; }

// Quantized types
typedef ap_int<8>  q8_t;
typedef ap_int<32> q32_t;

// Pack/unpack helpers (must match kernel little-endian layout)
static inline ap_uint<32> pack4(q8_t a0, q8_t a1, q8_t a2, q8_t a3) {
    ap_uint<32> w = 0;
    w.range(7,0)    = ap_uint<8>(a0);
    w.range(15,8)   = ap_uint<8>(a1);
    w.range(23,16)  = ap_uint<8>(a2);
    w.range(31,24)  = ap_uint<8>(a3);
    return w;
}
static inline void unpack4(ap_uint<32> w, q8_t &a0, q8_t &a1, q8_t &a2, q8_t &a3) {
    a0 = q8_t(ap_int<8>(w.range(7,0)));
    a1 = q8_t(ap_int<8>(w.range(15,8)));
    a2 = q8_t(ap_int<8>(w.range(23,16)));
    a3 = q8_t(ap_int<8>(w.range(31,24)));
}

// Kernel under test
extern "C" void conv1_preload_axis(
    hls::stream<axis_t> &act_s,
    hls::stream<axis_t> &out_s,
    hls::stream<axis_t> &param_s,
    int load_params,
    int req_shift,
    const q32_t req_m[C_OUT]
);

// Software reference implementing Conv2 + requant + ReLU + MaxPool
void reference_conv1(const std::vector<q8_t> &A,
                     const std::vector<q8_t> &W,
                     const std::vector<q32_t> &B,
                     const std::vector<q32_t> &REQ_M,
                     int req_shift,
                     std::vector<q8_t> &out)
{
    std::vector<q8_t> C(CONV_ELEMS);

    for (int y = 0; y < H_CONV; ++y) {
        for (int x = 0; x < W_CONV; ++x) {
            for (int co = 0; co < C_OUT; ++co) {
                q32_t acc = B[co];
                for (int ci = 0; ci < C_IN; ++ci) {
                    for (int ky = 0; ky < K; ++ky) {
                        for (int kx = 0; kx < K; ++kx) {
                            q8_t vin = A[idx_act(ci, y + ky, x + kx)];
                            q8_t wt  = W[idx_w(co, ci, ky, kx)];
                            acc += q32_t(vin) * q32_t(wt);
                        }
                    }
                }
                q32_t t = acc * REQ_M[co] + ((req_shift > 0) ? (q32_t(1) << (req_shift - 1)) : q32_t(0));
                q32_t s = (req_shift > 0) ? (t >> req_shift) : t;
                q8_t q  = (s > 127) ? q8_t(127) : (s < -128) ? q8_t(-128) : q8_t(s);
                if (q < q8_t(0)) q = q8_t(0);
                C[idx_c(co, y, x)] = q;
            }
        }
    }

    // MaxPool 2x2 stride 2
    for (int co = 0; co < C_OUT; ++co) {
        for (int y = 0; y < H_OUT; ++y) {
            for (int x = 0; x < W_OUT; ++x) {
                int y0 = y * S_POOL;
                int x0 = x * S_POOL;
                q8_t m = C[idx_c(co, y0,     x0    )];
                q8_t t = C[idx_c(co, y0,     x0 + 1)]; if (t > m) m = t;
                          t = C[idx_c(co, y0 + 1, x0    )]; if (t > m) m = t;
                          t = C[idx_c(co, y0 + 1, x0 + 1)]; if (t > m) m = t;
                out[idx_out(co, y, x)] = m;
            }
        }
    }
}

int main() {
    // Build deterministic int8 activations and weights, int32 bias
    std::vector<q8_t>  A(ACT_ELEMS);
    std::vector<q8_t>  W(W_ELEMS);
    std::vector<q32_t> B(B_ELEMS);

    for (int i = 0; i < ACT_ELEMS; ++i) A[i] = q8_t(((i % 9) - 4));        // range [-4,4]
    for (int i = 0; i < W_ELEMS;   ++i) W[i] = q8_t(((i % 7) - 3));        // range [-3,3]
    for (int i = 0; i < B_ELEMS;   ++i) B[i] = q32_t((i % 5) - 2);         // small biases

    // Quantization scales → fixed-point requant multipliers
    // For testbench, choose req_shift=8 and REQ_M all 64 → s ≈ acc * (64 / 256) = acc / 4
    int req_shift = 8;
    std::vector<q32_t> REQ_M(C_OUT, q32_t(64));

    // Prepare AXI streams
    hls::stream<axis_t> act_s, out_s, param_s;

    // Build parameter stream: weights then bias
    // Weights: pack 4 int8 per beat
    {
        int wi = 0;
        for (int i = 0; i < W_BEATS; ++i) {
            q8_t a0 = (wi < W_ELEMS) ? W[wi++] : q8_t(0);
            q8_t a1 = (wi < W_ELEMS) ? W[wi++] : q8_t(0);
            q8_t a2 = (wi < W_ELEMS) ? W[wi++] : q8_t(0);
            q8_t a3 = (wi < W_ELEMS) ? W[wi++] : q8_t(0);
            axis_t pkt;
            pkt.data = pack4(a0, a1, a2, a3);
            pkt.keep = 0xF;
            pkt.strb = 0x0;
            pkt.last = 0;
            param_s.write(pkt);
        }
    }
    // Bias: one int32 per beat
    for (int i = 0; i < B_BEATS; ++i) {
        axis_t pkt;
        pkt.data = ap_uint<32>(ap_int<32>(B[i]));
        pkt.keep = 0xF;
        pkt.strb = 0x0;
        pkt.last = 0;
        param_s.write(pkt);
    }

    // Build activation stream: pack 4 int8 per beat
    {
        int ai = 0;
        for (int i = 0; i < ACT_BEATS; ++i) {
            q8_t a0 = (ai < ACT_ELEMS) ? A[ai++] : q8_t(0);
            q8_t a1 = (ai < ACT_ELEMS) ? A[ai++] : q8_t(0);
            q8_t a2 = (ai < ACT_ELEMS) ? A[ai++] : q8_t(0);
            q8_t a3 = (ai < ACT_ELEMS) ? A[ai++] : q8_t(0);
            axis_t pkt;
            pkt.data = pack4(a0, a1, a2, a3);
            pkt.keep = 0xF;
            pkt.strb = 0x0;
            pkt.last = 0; // kernel doesn't rely on TLAST for parsing
            act_s.write(pkt);
        }
    }

    // Prepare AXI-Lite arguments (req_m array)
    q32_t req_m_arr[C_OUT];
    for (int co = 0; co < C_OUT; ++co) req_m_arr[co] = REQ_M[co];

    // Run kernel: load params and process one frame
    conv1_preload_axis(act_s, out_s, param_s, /*load_params=*/1, req_shift, req_m_arr);

    // Read outputs: expect OUT_BEATS beats; TLAST = 1 on last beat
    std::vector<q8_t> out_hw(OUT_ELEMS);
    {
        int oi = 0;
        for (int i = 0; i < OUT_BEATS; ++i) {
            if (out_s.empty()) {
                std::cerr << "ERROR: output stream underrun at beat " << i << "\n";
                return 1;
            }
            axis_t pkt = out_s.read();
            q8_t a0, a1, a2, a3;
            unpack4(pkt.data, a0, a1, a2, a3);
            if (oi < OUT_ELEMS) out_hw[oi++] = a0;
            if (oi < OUT_ELEMS) out_hw[oi++] = a1;
            if (oi < OUT_ELEMS) out_hw[oi++] = a2;
            if (oi < OUT_ELEMS) out_hw[oi++] = a3;

            if ((i == OUT_BEATS - 1 && pkt.last != 1) ||
                (i != OUT_BEATS - 1 && pkt.last == 1)) {
                std::cerr << "ERROR: TLAST protocol violation at beat " << i << "\n";
                return 1;
            }
        }
    }

    // Software reference
    std::vector<q8_t> out_ref(OUT_ELEMS, q8_t(0));
    reference_conv1(A, W, B, REQ_M, req_shift, out_ref);

    // Compare
    int mismatches = 0;
    for (int i = 0; i < OUT_ELEMS; ++i) {
        int hw = (int)out_hw[i];
        int rf = (int)out_ref[i];
        if (hw != rf) {
            if (++mismatches <= 20) {
                std::cout << "Mismatch at " << i << ": hw=" << hw << " ref=" << rf << "\n";
            }
        }
    }

    if (mismatches == 0) {
        std::cout << "PASS: All " << OUT_ELEMS << " int8 outputs match exactly\n";
        return 0;
    } else {
        std::cout << "FAIL: " << mismatches << " mismatches out of " << OUT_ELEMS << "\n";
        return 1;
    }
}