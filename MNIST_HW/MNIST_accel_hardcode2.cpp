#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <cstdint>

typedef ap_axiu<32,0,0,0> axis_t;

constexpr int C_IN   = 6;   // input channels from pool1
constexpr int H_IN   = 12;
constexpr int W_IN   = 12;

constexpr int C1_OUT = 16;  // conv2 filters
constexpr int K      = 5;

constexpr int H1     = H_IN - K + 1; // 8
constexpr int W1     = W_IN - K + 1; // 8

constexpr int S_POOL = 2;            // 2x2, stride 2
constexpr int H1P    = H1 / S_POOL;  // 4
constexpr int W1P    = W1 / S_POOL;  // 4

// Element counts
constexpr int IMG_ELEMS = C_IN * H_IN * W_IN;            // 864
constexpr int W_ELEMS   = C1_OUT * C_IN * K * K;         // 2400
constexpr int B_ELEMS   = C1_OUT;                        // 16
constexpr int CONV_ELEMS = C1_OUT * H1  * W1;            // 1024
constexpr int OUT_ELEMS  = C1_OUT * H1P * W1P;           // 256

// Index helpers
static inline int idx_img(int c, int y, int x)                 { return (c*H_IN + y)*W_IN + x; }
static inline int idx_w  (int co, int ci, int ky, int kx)      { return ((co*C_IN + ci)*K + ky)*K + kx; }
static inline int idx_c1 (int co, int y, int x)                { return (co*H1 + y)*W1 + x; }
static inline int idx_p1 (int co, int y, int x)                { return (co*H1P + y)*W1P + x; }

// Quantized types
typedef ap_int<8>  q8_t;
typedef ap_int<32> q32_t;

// Requantization parameters (tune these)
// y8 = clamp( (acc32 * M + rounding) >> shift )
#define  REQ_M  1    // multiplier (Q-format scale)
#define REQ_SHIFT 8  // right shift
static inline q8_t requantize(q32_t acc) {
    q32_t t = (acc * REQ_M) + ( (REQ_SHIFT > 0) ? (q32_t(1) << (REQ_SHIFT-1)) : q32_t(0) ); // round-to-nearest
    q32_t s = (REQ_SHIFT > 0) ? (t >> REQ_SHIFT) : t;
    // Saturate to int8
    if (s > 127)  return q8_t(127);
    if (s < -128) return q8_t(-128);
    return q8_t(s);
}

// Pack/unpack 4×int8 per 32-bit word (little-endian bytes)
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

extern "C" void lenet_conv2_relu_pool_axis(hls::stream<axis_t>& in_s,
                                           hls::stream<axis_t>& out_s)
{
#pragma HLS INTERFACE axis      port=in_s
#pragma HLS INTERFACE axis      port=out_s
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS BIND_OP op=mul impl=DSP
#pragma HLS ALLOCATION operation instances=mul limit=512

    // Local storage
    static q8_t  img[IMG_ELEMS];
    static q8_t  w   [W_ELEMS];
    static q32_t b   [B_ELEMS];

    static q8_t  conv_buf[CONV_ELEMS];
#pragma HLS BIND_STORAGE variable=conv_buf type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=img type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=w type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=b type=ram_2p impl=bram

    // 1) Read input payload: activations(int8, 4/beat) -> weights(int8, 4/beat) -> bias(int32, 1/beat)
    // Activations
    {
        const int beats = (IMG_ELEMS + 3) / 4;
        int idx = 0;
        for (int i = 0; i < beats; ++i) {
// #pragma HLS PIPELINE II=4
            axis_t pkt = in_s.read();
            q8_t a0, a1, a2, a3;
            unpack4(pkt.data, a0, a1, a2, a3);
            if (idx < IMG_ELEMS) img[idx++] = a0;
            if (idx < IMG_ELEMS) img[idx++] = a1;
            if (idx < IMG_ELEMS) img[idx++] = a2;
            if (idx < IMG_ELEMS) img[idx++] = a3;
        }
    }
    // Weights
    {
        const int beats = (W_ELEMS + 3) / 4;
        int idx = 0;
        for (int i = 0; i < beats; ++i) {
// #pragma HLS PIPELINE II=4
            axis_t pkt = in_s.read();
            q8_t a0, a1, a2, a3;
            unpack4(pkt.data, a0, a1, a2, a3);
            if (idx < W_ELEMS) w[idx++] = a0;
            if (idx < W_ELEMS) w[idx++] = a1;
            if (idx < W_ELEMS) w[idx++] = a2;
            if (idx < W_ELEMS) w[idx++] = a3;
        }
    }
    // Bias (int32)
    for (int i = 0; i < B_ELEMS; ++i) {
// #pragma HLS PIPELINE II=4
        axis_t pkt = in_s.read();
        // Interpret pkt.data as signed 32-bit bias
        b[i] = q32_t(ap_int<32>(pkt.data));
    }

    // 2) Conv2 + ReLU (int8 inputs/weights, int32 accumulate, int8 output)
    for (int y = 0; y < H1; ++y) {
        for (int x = 0; x < W1; ++x) {
// #pragma HLS PIPELINE II=4
            for (int co = 0; co < C1_OUT; ++co) {
                q32_t acc = b[co];
                for (int ci = 0; ci < C_IN; ++ci) {
                    for (int ky = 0; ky < K; ++ky) {
                        for (int kx = 0; kx < K; ++kx) {
                            q8_t vin_q = img[idx_img(ci, y + ky, x + kx)];
                            q8_t wgt_q = w[idx_w(co, ci, ky, kx)];
                            // Promote to 32-bit before multiply-accumulate
                            acc += q32_t(vin_q) * q32_t(wgt_q);
                        }
                    }
                }
                // Requantize to int8 and ReLU
                q8_t out_q = requantize(acc);
                if (out_q < q8_t(0)) out_q = q8_t(0);
                conv_buf[idx_c1(co, y, x)] = out_q;
            }
        }
    }

    // 3) MaxPool (2x2, stride 2) on int8 and stream out (pack 4 outputs per beat)
    int out_index = 0;
    q8_t pack_buf[4];
    int pack_cnt = 0;

    for (int co = 0; co < C1_OUT; ++co) {
        for (int y = 0; y < H1P; ++y) {
            for (int x = 0; x < W1P; ++x) {
// #pragma HLS PIPELINE II=4
                int y0 = y * S_POOL;
                int x0 = x * S_POOL;
                q8_t m = conv_buf[idx_c1(co, y0,     x0    )];
                q8_t t = conv_buf[idx_c1(co, y0,     x0 + 1)];
                if (t > m) m = t;
                t =        conv_buf[idx_c1(co, y0 + 1, x0    )];
                if (t > m) m = t;
                t =        conv_buf[idx_c1(co, y0 + 1, x0 + 1)];
                if (t > m) m = t;

                // Accumulate 4 pooled samples into one 32-bit word
                pack_buf[pack_cnt++] = m;
                if (pack_cnt == 4) {
                    ap_uint<32> word = pack4(pack_buf[0], pack_buf[1], pack_buf[2], pack_buf[3]);
                    axis_t pkt;
                    pkt.data = word;
                    pkt.keep = 0xF;       // all bytes valid
                    pkt.strb = 0x0;
                    // TLAST on last beat
                    int beats_left = ((OUT_ELEMS - out_index) + 3) / 4;
                    pkt.last = (beats_left == 1) ? 1 : 0;
                    out_s.write(pkt);
                    out_index += 4;
                    pack_cnt = 0;
                }
            }
        }
    }
    // Flush remaining (<4) outputs if OUT_ELEMS not multiple of 4
    if (pack_cnt > 0) {
        q8_t a0 = (pack_cnt > 0) ? pack_buf[0] : q8_t(0);
        q8_t a1 = (pack_cnt > 1) ? pack_buf[1] : q8_t(0);
        q8_t a2 = (pack_cnt > 2) ? pack_buf[2] : q8_t(0);
        q8_t a3 = (pack_cnt > 3) ? pack_buf[3] : q8_t(0);
        ap_uint<32> word = pack4(a0, a1, a2, a3);
        axis_t pkt;
        pkt.data = word;
        pkt.keep = 0xF;
        pkt.strb = 0x0;
        pkt.last = 1; // last beat
        out_s.write(pkt);
    }
}