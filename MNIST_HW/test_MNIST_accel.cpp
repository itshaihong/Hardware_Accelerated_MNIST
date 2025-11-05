// tb_cnn_accel.cpp
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstring>
#include "MNIST_accel.hpp"
#include <cassert>

// Reference CPU implementation (matches cnn_accel behavior)
static inline qint8 clamp_int8_ref(int v) {
    if (v > 127) return 127;
    if (v < -128) return -128;
    return (qint8)v;
}

static qint8 ref_conv5x5_pixel(
    const std::vector<qint8>& in_q,
    const std::vector<qint8>& w_q,
    qint16 bias,
    int Cin, int H, int W,
    int co, int ho, int wo,
    int pad,
    float scale_S,
    int shift
) {
    qint16 acc = bias;
    for (int ci = 0; ci < Cin; ++ci) {
        const int w_base = ((co * Cin) + ci) * 25; // [co][ci][5][5]
        for (int kh = 0; kh < 5; ++kh) {
            const int h_in = ho + kh - pad;
            for (int kw = 0; kw < 5; ++kw) {
                const int w_in = wo + kw - pad;
                qint8 iv = 0;
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    const int in_idx = (ci * H + h_in) * W + w_in;
                    iv = in_q[in_idx];
                }
                const int w_idx = w_base + kh * 5 + kw;
                const qint8 wv = w_q[w_idx];
                acc += (qint16)iv * (qint16)wv;
            }
        }
    }
    if (acc < 0) acc = 0; // ReLU
    int scaled = (int)(scale_S * (float)acc);
    if (shift > 0) scaled >>= shift;
    return clamp_int8_ref(scaled);
}

static void ref_run(
    const std::vector<qint8>& in_q,
    const std::vector<qint8>& w_q,
    const std::vector<qint16>& b_q,
    std::vector<qint8>& out_q,
    int Cin, int H, int W, int Cout, int pad, int pool, float scale_S, int shift
) {
    const int Hout_conv = H + 2*pad - 5 + 1;
    const int Wout_conv = W + 2*pad - 5 + 1;

    if (pool == POOL_NONE) {
        out_q.assign(Cout * Hout_conv * Wout_conv, 0);
        for (int co = 0; co < Cout; ++co) {
            for (int ho = 0; ho < Hout_conv; ++ho) {
                for (int wo = 0; wo < Wout_conv; ++wo) {
                    qint8 y = ref_conv5x5_pixel(in_q, w_q, b_q[co], Cin, H, W, co, ho, wo, pad, scale_S, shift);
                    const int out_idx = (co * Hout_conv + ho) * Wout_conv + wo;
                    assert(out_idx >= 0 && (size_t)out_idx < out_q.size());
                    out_q[out_idx] = y;
                }
            }
        }
    } else { // POOL_MAX2x2
        const int Hout = Hout_conv / 2;
        const int Wout = Wout_conv / 2;
        out_q.assign(Cout * Hout * Wout, 0);
        for (int co = 0; co < Cout; ++co) {
            for (int ho2 = 0; ho2 < Hout; ++ho2) {
                const int ho = ho2 * 2;
                for (int wo2 = 0; wo2 < Wout; ++wo2) {
                    const int wo = wo2 * 2;
                    qint8 a = ref_conv5x5_pixel(in_q, w_q, b_q[co], Cin, H, W, co, ho,   wo,   pad, scale_S, shift);
                    qint8 b = ref_conv5x5_pixel(in_q, w_q, b_q[co], Cin, H, W, co, ho,   wo+1, pad, scale_S, shift);
                    qint8 c = ref_conv5x5_pixel(in_q, w_q, b_q[co], Cin, H, W, co, ho+1, wo,   pad, scale_S, shift);
                    qint8 d = ref_conv5x5_pixel(in_q, w_q, b_q[co], Cin, H, W, co, ho+1, wo+1, pad, scale_S, shift);
                    qint8 m = a; if (b > m) m = b; if (c > m) m = c; if (d > m) m = d;
                    const int out_idx = (co * Hout + ho2) * Wout + wo2;
                    assert(out_idx >= 0 && (size_t)out_idx < out_q.size());
                    out_q[out_idx] = m;
                }
            }
        }
    }
}

static void gen_random(std::vector<qint8>& v) {
    for (auto& x : v) {
        int r = (std::rand() % 255) - 128; // [-128, 126]
        x = (qint8)r;
    }
}
static void gen_random_bias(std::vector<qint16>& v) {
    for (auto& x : v) {
        int r = (std::rand() % 1024) - 512; // [-512, 511]
        x = (qint16)r;
    }
}

std::vector<int> status_buf(1, 0);

static int run_one_case(
    int Cin, int H, int W, int Cout, int pad, int pool,
    float scale_S, int shift, bool verbose = true
) {
    const int Hout_conv = H + 2*pad - 5 + 1;
    const int Wout_conv = W + 2*pad - 5 + 1;
    if (Hout_conv <= 0 || Wout_conv <= 0) {
        std::cerr << "Invalid output size for given H/W/pad\n";
        return -1;
    }
    const int Hout = (pool == POOL_NONE) ? Hout_conv : (Hout_conv / 2);
    const int Wout = (pool == POOL_NONE) ? Wout_conv : (Wout_conv / 2);

    std::vector<qint8> in_q(Cin * H * W);
    std::vector<qint8> w_q(Cout * Cin * 5 * 5);
    std::vector<qint16> b_q(Cout);
    std::vector<qint8> out_q_hw(Cout * Hout * Wout, 0);
    std::vector<qint8> out_q_ref(Cout * Hout * Wout, 0);

    gen_random(in_q);
    gen_random(w_q);
    gen_random_bias(b_q);

    // Call HLS kernel
    status_buf[0] = 0;
    cnn_accel(in_q.data(), w_q.data(), b_q.data(), out_q_hw.data(),
              Cin, H, W, Cout, pad, pool, scale_S, shift, status_buf.data());
    while (status_buf[0] != 1) {
    }

    // Reference
    ref_run(in_q, w_q, b_q, out_q_ref, Cin, H, W, Cout, pad, pool, scale_S, shift);

    // Compare
    int mismatches = 0;
    for (size_t i = 0; i < Cout * Hout * Wout; ++i) {
        if (out_q_hw[i] != out_q_ref[i]) {
            if (mismatches < 10 && verbose) {
                std::cerr << "Mismatch at " << i
                          << " hw=" << (int)out_q_hw[i]
                          << " ref=" << (int)out_q_ref[i] << "\n";
            }
            ++mismatches;
        }
    }
    if (verbose) {
        std::cout << "Case Cin=" << Cin << " H=" << H << " W=" << W
                  << " Cout=" << Cout << " pad=" << pad
                  << " pool=" << (pool ? "MAX2x2" : "NONE")
                  << " -> out " << Hout << "x" << Wout
                  << " mismatches=" << mismatches << " / " << out_q_ref.size()
                  << (mismatches == 0 ? " [PASS]\n" : " [FAIL]\n");
    }
    return mismatches;
}



int main() {
    std::srand(0xC0FFEE);

    int total_fail = 0;

    // Test 1: LeNet Conv1-like: Cin=1, H=W=28, Cout=6, pad=2, pool=on
    total_fail += (run_one_case(1, 28, 28, 6, 2, POOL_MAX2x2, 1.0f, 0) != 0);

    // Test 2: LeNet Conv2-like: Cin=6, H=W=14, Cout=16, pad=0, pool=on
    total_fail += (run_one_case(6, 14, 14, 16, 0, POOL_MAX2x2, 1.0f, 0) != 0);

    // Test 3: No pooling path
    // total_fail += (run_one_case(3, 16, 16, 8, 1, POOL_NONE, 1.0f, 0) != 0);

    // Test 4: With scaling and shift (quantization)
    // total_fail += (run_one_case(4, 20, 18, 7, 2, POOL_NONE, 0.5f, 1) != 0);

    if (total_fail == 0) {
        std::cout << "All tests PASS\n";
        return 0;
    } else {
        std::cerr << "Some tests FAILED (" << total_fail << ")\n";
        return 1;
    }
}