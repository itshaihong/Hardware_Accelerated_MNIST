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

typedef ap_axis<32,0,0,0> axis_t;

// LeNet-5 Conv2 fixed sizes (must match kernel)
constexpr int C_IN   = 6;
constexpr int H_IN   = 14;
constexpr int W_IN   = 14;
constexpr int C_OUT  = 16;
constexpr int K      = 5;

constexpr int H_CONV = H_IN - K + 1; // 8
constexpr int W_CONV = W_IN - K + 1; // 8

constexpr int S_POOL = 2;
constexpr int H_OUT  = H_CONV / S_POOL; // 4
constexpr int W_OUT  = W_CONV / S_POOL; // 4

constexpr int ACT_ELEMS   = C_IN * H_IN    * W_IN;     // 864
constexpr int W_ELEMS     = C_OUT * C_IN * K * K;      // 2400
constexpr int B_ELEMS     = C_OUT;                     // 16
constexpr int CONV_ELEMS  = C_OUT * H_CONV * W_CONV;   // 1024
constexpr int OUT_ELEMS   = C_OUT * H_OUT  * W_OUT;    // 256

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


// Kernel under test
void conv2_preload_axis(hls::stream<axis_t>& act_s, //last activation stream
                    hls::stream<axis_t>& wb_s, //weight and bias stream
                    hls::stream<axis_t>& out_s);

// Reference software (same algorithm)
void reference_conv1_relu_pool(const std::vector<int>& img,
                               const std::vector<int>& w,
                               const std::vector<int>& b,
                               std::vector<int>& out)
{
    std::vector<int> conv(CONV_ELEMS);
    // Conv + ReLU
    for (int y = 0; y < H_CONV; ++y) {
        for (int x = 0; x < W_CONV; ++x) {
            for (int co = 0; co < C_OUT; ++co) {
                int acc = 0;
                for (int ci = 0; ci < C_IN; ++ci) {
                    for (int ky = 0; ky < K; ++ky) {
                        for (int kx = 0; kx < K; ++kx) {
                            int vin = img[idx_act(ci, y + ky, x + kx)];
                            int wgt = w[idx_w(co, ci, ky, kx)];
                            acc += (vin * wgt);
                        }
                    }
                }
                acc = acc >> 8;
                acc = acc + b[co];
                if (acc < (int)0) acc = (int)0;
                conv[idx_c(co, y, x)] = acc;
            }
        }
    }
    // MaxPool 2x2 stride 2
    for (int y = 0; y < H_OUT; ++y) {
        for (int x = 0; x < W_OUT; ++x) {
            for (int co = 0; co < C_OUT; ++co) {
                int y0 = y * S_POOL, x0 = x * S_POOL;
                int m = conv[idx_c(co, y0,     x0    )];
                int t = conv[idx_c(co, y0,     x0 + 1)];
                if (t > m) m = t;
                t =        conv[idx_c(co, y0 + 1, x0    )];
                if (t > m) m = t;
                t =        conv[idx_c(co, y0 + 1, x0 + 1)];
                if (t > m) m = t;
                out[idx_out(co, y, x)] = m;
                if (idx_out(co, y, x) <= 2){
                }
            }
        }
    }
}

int main() {
    // Build deterministic int8 activations and weights, int32 bias
    std::vector<int> img(ACT_ELEMS);
    std::vector<int> w(W_ELEMS);
    std::vector<int> b(B_ELEMS);
    for (int i = 0; i < ACT_ELEMS; ++i) img[i] = i % 32;
    for (int i = 0; i < W_ELEMS;   ++i) w[i]   = 5;
    for (int i = 0; i < B_ELEMS;   ++i) b[i]   = 1;

    std::vector<int> payload;
    payload.reserve(W_ELEMS+B_ELEMS);
    payload.insert(payload.end(), w.begin(),  w.end());
    payload.insert(payload.end(), b.begin(),  b.end());
    assert((int)payload.size() == W_ELEMS+B_ELEMS);

    // Prepare AXI streams
    hls::stream<axis_t> act_s, out_s, param_s;

    // Drive input stream (TLAST on final beat)
    for (int i = 0; i < ACT_ELEMS; ++i) {
        axis_t pkt{};
        pkt.data = img[i];
        pkt.keep = -1;      // all bytes valid for 32-bit
        pkt.last = (i == ACT_ELEMS - 1) ? 1 : 0;
        act_s.write(pkt);
    }

    for (int i = 0; i < W_ELEMS+B_ELEMS; ++i) {
        axis_t pkt{};
        pkt.data = payload[i];
        pkt.keep = -1;      // all bytes valid for 32-bit
        pkt.last = (i == ACT_ELEMS - 1) ? 1 : 0;
        param_s.write(pkt);
    }


    // Run kernel: load params and process one frame
    conv2_preload_axis(act_s, param_s, out_s);

    // Read outputs (expect OUT_ELEMS = 864), TLAST on last beat
    std::vector<int> out_hw(OUT_ELEMS);
    for (int i = 0; i < OUT_ELEMS; ++i) {
        if (out_s.empty()) {
            std::cerr << "ERROR: output underrun at i=" << i << "\n";
            return 1;
        }
        axis_t pkt = out_s.read();
        out_hw[i] = pkt.data;
        if ((i == OUT_ELEMS - 1 && pkt.last != 1) ||
            (i != OUT_ELEMS - 1 && pkt.last == 1)) {
            std::cerr << "ERROR: TLAST protocol violation at i=" << i << "\n";
            return 1;
        }
    }

    // Software reference and compare
    std::vector<int> out_ref(OUT_ELEMS, 0);
    reference_conv1_relu_pool(img, w, b, out_ref);

    int mismatches = 0;
    const float tol = 1e-3f;
    for (int i = 0; i < OUT_ELEMS; ++i) {
        float a = out_hw[i];
        float r = out_ref[i];
        float diff = std::fabs(a - r);
        if (i<=256){
                            std::cout << "at " << i
                          << ": hw=" << a << " ref=" << r
                          << " diff=" << diff << "\n";
        }
        if (diff > tol || !std::isfinite(a) || !std::isfinite(r)) {
            if (++mismatches <= 20) {
                std::cout << "Mismatch at " << i
                          << ": hw=" << a << " ref=" << r
                          << " diff=" << diff << "\n";
            }
        }
    }

    if (mismatches == 0) {
        std::cout << "PASS: All " << OUT_ELEMS << " outputs match within tol=" << tol << "\n";
        return 0;
    } else {
        std::cout << "FAIL: " << mismatches << " mismatches out of " << OUT_ELEMS << "\n";
        return 1;
    }
}