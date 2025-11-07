// tb_lenet_conv2_relu_pool_axis.cpp
// Testbench for int8-quantized lenet_conv2_relu_pool_axis kernel.
// Payload order: Activations (int8, 4 per 32-bit beat) -> Weights (int8, 4 per beat) -> Bias (int32, 1 per beat)

#include <iostream>
#include <vector>
#include <cstdint>
#include <cassert>

#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>

typedef ap_axiu<32,0,0,0> axis_t;

constexpr int C_IN   = 6;
constexpr int H_IN   = 12;
constexpr int W_IN   = 12;

constexpr int C1_OUT = 16;
constexpr int K      = 5;

constexpr int H1     = H_IN - K + 1; // 8
constexpr int W1     = W_IN - K + 1; // 8

constexpr int S_POOL = 2;
constexpr int H1P    = H1 / S_POOL;  // 4
constexpr int W1P    = W1 / S_POOL;  // 4

constexpr int IMG_ELEMS   = C_IN * H_IN * W_IN;            // 864
constexpr int W_ELEMS     = C1_OUT * C_IN * K * K;         // 2400
constexpr int B_ELEMS     = C1_OUT;                        // 16
constexpr int CONV_ELEMS  = C1_OUT * H1  * W1;             // 1024
constexpr int OUT_ELEMS   = C1_OUT * H1P * W1P;            // 256

// Index helpers (match kernel)
static inline int idx_img(int c, int y, int x)                 { return (c*H_IN + y)*W_IN + x; }
static inline int idx_w  (int co, int ci, int ky, int kx)      { return ((co*C_IN + ci)*K + ky)*K + kx; }
static inline int idx_c1 (int co, int y, int x)                { return (co*H1 + y)*W1 + x; }
static inline int idx_p1 (int co, int y, int x)                { return (co*H1P + y)*W1P + x; }

// Quantized types and requantization (must match kernel)
typedef ap_int<8>  q8_t;
typedef ap_int<32> q32_t;

#define REQ_M  1 // same as kernel
#define REQ_SHIFT 8
static inline q8_t requantize(q32_t acc) {
    q32_t t = (acc * REQ_M) + ((REQ_SHIFT > 0) ? (q32_t(1) << (REQ_SHIFT-1)) : q32_t(0));
    q32_t s = (REQ_SHIFT > 0) ? (t >> REQ_SHIFT) : t;
    if (s > 127)  return q8_t(127);
    if (s < -128) return q8_t(-128);
    return q8_t(s);
}

// Pack/unpack 4×int8 into 32-bit word (must match kernel little-endian layout)
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
extern "C" void lenet_conv2_relu_pool_axis(hls::stream<axis_t>& in_s,
                                           hls::stream<axis_t>& out_s);

// Software reference (int8 conv + int8 ReLU + 2x2 maxpool)
// Uses the same quantization and indexing as the kernel.
void reference_conv2_relu_pool(const std::vector<q8_t>& img,
                               const std::vector<q8_t>& w,
                               const std::vector<q32_t>& b,
                               std::vector<q8_t>& out)
{
    std::vector<q8_t> conv(CONV_ELEMS);

    for (int y = 0; y < H1; ++y) {
        for (int x = 0; x < W1; ++x) {
            for (int co = 0; co < C1_OUT; ++co) {
                q32_t acc = b[co];
                for (int ci = 0; ci < C_IN; ++ci) {
                    for (int ky = 0; ky < K; ++ky) {
                        for (int kx = 0; kx < K; ++kx) {
                            q8_t vin = img[idx_img(ci, y + ky, x + kx)];
                            q8_t wt  = w[idx_w(co, ci, ky, kx)];
                            acc += q32_t(vin) * q32_t(wt);
                        }
                    }
                }
                q8_t q = requantize(acc);
                if (q < q8_t(0)) q = q8_t(0);
                conv[idx_c1(co, y, x)] = q;
            }
        }
    }

    for (int co = 0; co < C1_OUT; ++co) {
        for (int y = 0; y < H1P; ++y) {
            for (int x = 0; x < W1P; ++x) {
                int y0 = y * S_POOL;
                int x0 = x * S_POOL;
                q8_t m = conv[idx_c1(co, y0,     x0    )];
                q8_t t = conv[idx_c1(co, y0,     x0 + 1)];
                if (t > m) m = t;
                t =        conv[idx_c1(co, y0 + 1, x0    )];
                if (t > m) m = t;
                t =        conv[idx_c1(co, y0 + 1, x0 + 1)];
                if (t > m) m = t;
                out[idx_p1(co, y, x)] = m;
            }
        }
    }
}

int main() {
    // Create deterministic int8 activations and weights, and int32 bias
    std::vector<q8_t>  img(IMG_ELEMS);
    std::vector<q8_t>  w(W_ELEMS);
    std::vector<q32_t> b(B_ELEMS);

    for (int i = 0; i < IMG_ELEMS; ++i) img[i] = q8_t(((i % 9) - 4));         // small symmetric values
    for (int i = 0; i < W_ELEMS;   ++i) w[i]   = q8_t(((i % 7) - 3));         // small symmetric values
    for (int i = 0; i < B_ELEMS;   ++i) b[i]   = q32_t((i % 3) - 1);          // small biases

    // Build AXI stream payload: pack 4 int8 per 32-bit word for IMG and W; 1 int32 per word for B
    hls::stream<axis_t> in_s, out_s;

    // Activations
    {
        int idx = 0;
        const int beats = (IMG_ELEMS + 3) / 4;
        for (int i = 0; i < beats; ++i) {
            q8_t a0 = (idx < IMG_ELEMS) ? img[idx++] : q8_t(0);
            q8_t a1 = (idx < IMG_ELEMS) ? img[idx++] : q8_t(0);
            q8_t a2 = (idx < IMG_ELEMS) ? img[idx++] : q8_t(0);
            q8_t a3 = (idx < IMG_ELEMS) ? img[idx++] : q8_t(0);
            axis_t pkt;
            pkt.data = pack4(a0, a1, a2, a3);
            pkt.keep = 0xF;
            pkt.strb = 0x0;
            pkt.last = 0;
            in_s.write(pkt);
        }
    }
    // Weights
    {
        int idx = 0;
        const int beats = (W_ELEMS + 3) / 4;
        for (int i = 0; i < beats; ++i) {
            q8_t a0 = (idx < W_ELEMS) ? w[idx++] : q8_t(0);
            q8_t a1 = (idx < W_ELEMS) ? w[idx++] : q8_t(0);
            q8_t a2 = (idx < W_ELEMS) ? w[idx++] : q8_t(0);
            q8_t a3 = (idx < W_ELEMS) ? w[idx++] : q8_t(0);
            axis_t pkt;
            pkt.data = pack4(a0, a1, a2, a3);
            pkt.keep = 0xF;
            pkt.strb = 0x0;
            pkt.last = 0;
            in_s.write(pkt);
        }
    }
    // Bias
    for (int i = 0; i < B_ELEMS; ++i) {
        axis_t pkt;
        pkt.data = ap_uint<32>(ap_int<32>(b[i]));
        pkt.keep = 0xF;
        pkt.strb = 0x0;
        // TLAST not required by kernel; leave 0
        pkt.last = 0;
        in_s.write(pkt);
    }

    // Call kernel
    lenet_conv2_relu_pool_axis(in_s, out_s);

    // Read outputs: expect OUT_ELEMS int8 packed 4 per beat
    std::vector<q8_t> out_hw(OUT_ELEMS);
    {
        int idx = 0;
        const int beats = (OUT_ELEMS + 3) / 4; // should be 64
        for (int i = 0; i < beats; ++i) {
            if (out_s.empty()) {
                std::cerr << "ERROR: output stream underrun at beat " << i << "\n";
                return 1;
            }
            axis_t pkt = out_s.read();
            q8_t a0, a1, a2, a3;
            unpack4(pkt.data, a0, a1, a2, a3);
            if (idx < OUT_ELEMS) out_hw[idx++] = a0;
            if (idx < OUT_ELEMS) out_hw[idx++] = a1;
            if (idx < OUT_ELEMS) out_hw[idx++] = a2;
            if (idx < OUT_ELEMS) out_hw[idx++] = a3;

            // TLAST should be 1 on last beat
            if ((i == beats - 1 && pkt.last != 1) || (i != beats - 1 && pkt.last == 1)) {
                std::cerr << "ERROR: TLAST protocol violation at beat " << i << "\n";
                return 1;
            }
        }
    }

    // Software reference and compare
    std::vector<q8_t> out_ref(OUT_ELEMS, q8_t(0));
    reference_conv2_relu_pool(img, w, b, out_ref);

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