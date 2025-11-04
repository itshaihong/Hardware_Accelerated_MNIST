#include "MNIST_accel.hpp"

static inline qint8 clamp_int8(int v) {
    if (v > 127) return 127;
    if (v < -128) return -128;
    return (qint8)v;
}

// Compute one conv+ReLU+requantized pixel for given (co, ho, wo)
static qint8 conv5x5_pixel(
    const qint8* in_q,
    const qint8* w_q,
    qint16 bias,
    int Cin, int H, int W,
    int co, int ho, int wo,
    int pad,
    float scale_S,
    int shift
) {
#pragma HLS INLINE
    // Map output position to input window top-left: (ho - pad, wo - pad)
    qint16 acc = bias;
    for (int ci = 0; ci < Cin; ++ci) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=16
        const int w_base = ((co * Cin) + ci) * 25; // [co][ci][5][5]
        for (int kh = 0; kh < 5; ++kh) {
#pragma HLS UNROLL
            const int h_in = ho + kh - pad;
            for (int kw = 0; kw < 5; ++kw) {
#pragma HLS UNROLL
                const int w_in = wo + kw - pad;
                qint8 iv = 0;
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    const int in_idx = (ci * H + h_in) * W + w_in;
                    iv = in_q[in_idx];
                }
                const int w_idx = w_base + kh * 5 + kw;
                qint8 wv = w_q[w_idx];
                acc += (qint16)iv * (qint16)wv;
            }
        }
    }
    // ReLU
    if (acc < 0) acc = 0;
    // Requantize to int8
    int scaled = (int)(scale_S * (float)acc);
    if (shift > 0) scaled >>= shift;
    return clamp_int8(scaled);
}

extern "C" {
void cnn_accel(
    const qint8*  in_q,
    const qint8*  w_q,
    const qint16* b_q,
    qint8*        out_q,
    int Cin,
    int H,
    int W,
    int Cout,
    int pad,
    int pool,
    float scale_S,
    int shift
) {
#pragma HLS INTERFACE m_axi     port=in_q   offset=slave bundle=gmem0 depth=65536
#pragma HLS INTERFACE m_axi     port=w_q    offset=slave bundle=gmem1 depth=65536
#pragma HLS INTERFACE m_axi     port=b_q    offset=slave bundle=gmem2 depth=1024
#pragma HLS INTERFACE m_axi     port=out_q  offset=slave bundle=gmem3 depth=65536

#pragma HLS INTERFACE s_axilite port=in_q    bundle=control
#pragma HLS INTERFACE s_axilite port=w_q     bundle=control
#pragma HLS INTERFACE s_axilite port=b_q     bundle=control
#pragma HLS INTERFACE s_axilite port=out_q   bundle=control
#pragma HLS INTERFACE s_axilite port=Cin     bundle=control
#pragma HLS INTERFACE s_axilite port=H       bundle=control
#pragma HLS INTERFACE s_axilite port=W       bundle=control
#pragma HLS INTERFACE s_axilite port=Cout    bundle=control
#pragma HLS INTERFACE s_axilite port=pad     bundle=control
#pragma HLS INTERFACE s_axilite port=pool    bundle=control
#pragma HLS INTERFACE s_axilite port=scale_S bundle=control
#pragma HLS INTERFACE s_axilite port=shift   bundle=control
#pragma HLS INTERFACE s_axilite port=return  bundle=control

    const int Hout_conv = H + 2*pad - 5 + 1;
    const int Wout_conv = W + 2*pad - 5 + 1;

    if (pool == POOL_NONE) {
        // Direct conv output
        for (int co = 0; co < Cout; ++co) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=32
            for (int ho = 0; ho < Hout_conv; ++ho) {
                for (int wo = 0; wo < Wout_conv; ++wo) {
#pragma HLS PIPELINE II=1
                    qint8 y = conv5x5_pixel(in_q, w_q, b_q[co], Cin, H, W, co, ho, wo, pad, scale_S, shift);
                    const int out_idx = (co * Hout_conv + ho) * Wout_conv + wo;
                    out_q[out_idx] = y;
                }
            }
        }
    } else { // POOL_MAX2x2
        const int Hout = Hout_conv / 2;
        const int Wout = Wout_conv / 2;
        for (int co = 0; co < Cout; ++co) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=32
            for (int ho2 = 0; ho2 < Hout; ++ho2) {
                const int ho = ho2 * 2;
                for (int wo2 = 0; wo2 < Wout; ++wo2) {
#pragma HLS PIPELINE II=1
                    const int wo = wo2 * 2;
                    qint8 a = conv5x5_pixel(in_q, w_q, b_q[co], Cin, H, W, co, ho,   wo,   pad, scale_S, shift);
                    qint8 b = conv5x5_pixel(in_q, w_q, b_q[co], Cin, H, W, co, ho,   wo+1, pad, scale_S, shift);
                    qint8 c = conv5x5_pixel(in_q, w_q, b_q[co], Cin, H, W, co, ho+1, wo,   pad, scale_S, shift);
                    qint8 d = conv5x5_pixel(in_q, w_q, b_q[co], Cin, H, W, co, ho+1, wo+1, pad, scale_S, shift);
                    qint8 m = a; if (b > m) m = b; if (c > m) m = c; if (d > m) m = d;
                    const int out_idx = (co * Hout + ho2) * Wout + wo2;
                    out_q[out_idx] = m;
                }
            }
        }
    }
}
}