// tb_lenet_conv1_relu_pool_axis.cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <cstdint>
#include <cstring>

using data_t = float;
typedef ap_axiu<32,0,0,0> axis_t;

constexpr int C_IN = 1;
constexpr int H_IN = 28;
constexpr int W_IN = 28;

constexpr int C1_OUT = 6;
constexpr int K      = 5;
constexpr int H1     = H_IN - K + 1; // 24
constexpr int W1     = W_IN - K + 1; // 24

constexpr int S_POOL = 2;
constexpr int H1P    = H1 / S_POOL;  // 12
constexpr int W1P    = W1 / S_POOL;  // 12

constexpr int IMG_ELEMS = C_IN * H_IN * W_IN;            // 784
constexpr int W_ELEMS   = C1_OUT * C_IN * K * K;         // 150
constexpr int B_ELEMS   = C1_OUT;                        // 6
constexpr int IN_TOTAL  = IMG_ELEMS + W_ELEMS + B_ELEMS; // 940

constexpr int CONV_ELEMS = C1_OUT * H1  * W1;            // 3456
constexpr int OUT_ELEMS  = C1_OUT * H1P * W1P;           // 864

static inline int idx_img(int c, int y, int x)            { return (c*H_IN + y)*W_IN + x; }
static inline int idx_w  (int co, int ci, int ky, int kx) { return ((co*C_IN + ci)*K + ky)*K + kx; }
static inline int idx_c1 (int co, int y, int x)           { return (co*H1 + y)*W1 + x; }
static inline int idx_p1 (int co, int y, int x)           { return (co*H1P + y)*W1P + x; }


static inline ap_uint<32> f2u(float f) {
    uint32_t tmp;
    ::memcpy(&tmp, &f, sizeof(tmp));
    return ap_uint<32>(tmp); // ap_uint has a constructor from integral types
}

static inline float u2f(ap_uint<32> u) {
    uint32_t tmp = static_cast<uint32_t>(u); // extract bits to plain integer
    float f;
    ::memcpy(&f, &tmp, sizeof(f));
    return f;
}

extern "C" void lenet_conv1_relu_pool_axis(hls::stream<axis_t>& in_s,
                                           hls::stream<axis_t>& out_s);

// Reference software (same algorithm)
void reference_conv1_relu_pool(const std::vector<data_t>& img,
                               const std::vector<data_t>& w,
                               const std::vector<data_t>& b,
                               std::vector<data_t>& out)
{
    std::vector<data_t> conv(CONV_ELEMS);
    // Conv + ReLU
    for (int y = 0; y < H1; ++y) {
        for (int x = 0; x < W1; ++x) {
            for (int co = 0; co < C1_OUT; ++co) {
                data_t acc = b[co];
                for (int ci = 0; ci < C_IN; ++ci) {
                    for (int ky = 0; ky < K; ++ky) {
                        for (int kx = 0; kx < K; ++kx) {
                            data_t vin = img[idx_img(ci, y + ky, x + kx)];
                            data_t wgt = w[idx_w(co, ci, ky, kx)];
                            acc += vin * wgt;
                        }
                    }
                }
                if (acc < (data_t)0) acc = (data_t)0;
                conv[idx_c1(co, y, x)] = acc;
            }
        }
    }
    // MaxPool 2x2 stride 2
    for (int y = 0; y < H1P; ++y) {
        for (int x = 0; x < W1P; ++x) {
            for (int co = 0; co < C1_OUT; ++co) {
                int y0 = y * S_POOL, x0 = x * S_POOL;
                data_t m = conv[idx_c1(co, y0,     x0    )];
                data_t t = conv[idx_c1(co, y0,     x0 + 1)];
                if (t > m) m = t;
                t =        conv[idx_c1(co, y0 + 1, x0    )];
                if (t > m) m = t;
                t =        conv[idx_c1(co, y0 + 1, x0 + 1)];
                if (t > m) m = t;
                out[idx_p1(co, y, x)] = m;
                if (idx_p1(co, y, x) <= 2){
                }
            }
        }
    }
}

int main() {
    // Prepare deterministic inputs
    std::vector<data_t> img(IMG_ELEMS);
    std::vector<data_t> w(W_ELEMS);
    std::vector<data_t> b(B_ELEMS);
    for (int i = 0; i < IMG_ELEMS; ++i) img[i] = ((i % 13) - 6) * 0.05f;
    for (int i = 0; i < W_ELEMS;   ++i) w[i]   = ((i % 7)  - 3) * 0.02f;
    for (int i = 0; i < B_ELEMS;   ++i) b[i]   = 0.01f * (i % 5);

    

    // Build payload = IMG | W | B
    std::vector<data_t> payload;
    payload.reserve(IN_TOTAL);
    payload.insert(payload.end(), img.begin(), img.end());
    payload.insert(payload.end(), w.begin(),  w.end());
    payload.insert(payload.end(), b.begin(),  b.end());
    assert((int)payload.size() == IN_TOTAL);

    hls::stream<axis_t> in_s, out_s;

    // Drive input stream (TLAST on final beat)
    for (int i = 0; i < IN_TOTAL; ++i) {
        axis_t pkt{};
        pkt.data = f2u(payload[i]);
        pkt.keep = -1;      // all bytes valid for 32-bit
        pkt.last = (i == IN_TOTAL - 1) ? 1 : 0;
        in_s.write(pkt);
    }

    // Call kernel
    lenet_conv1_relu_pool_axis(in_s, out_s);

    // Read outputs (expect OUT_ELEMS = 864), TLAST on last beat
    std::vector<data_t> out_hw(OUT_ELEMS);
    for (int i = 0; i < OUT_ELEMS; ++i) {
        if (out_s.empty()) {
            std::cerr << "ERROR: output underrun at i=" << i << "\n";
            return 1;
        }
        axis_t pkt = out_s.read();
        out_hw[i] = u2f(pkt.data);
        if ((i == OUT_ELEMS - 1 && pkt.last != 1) ||
            (i != OUT_ELEMS - 1 && pkt.last == 1)) {
            std::cerr << "ERROR: TLAST protocol violation at i=" << i << "\n";
            return 1;
        }
    }

    // Software reference and compare
    std::vector<data_t> out_ref(OUT_ELEMS, 0);
    reference_conv1_relu_pool(img, w, b, out_ref);

    int mismatches = 0;
    const float tol = 1e-3f;
    for (int i = 0; i < OUT_ELEMS; ++i) {
        float a = out_hw[i];
        float r = out_ref[i];
        float diff = std::fabs(a - r);
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