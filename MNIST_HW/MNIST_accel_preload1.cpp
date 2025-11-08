// conv2_int8_axilite_axis_fixed_tlast.cpp
#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <cstdint>

typedef ap_axiu<32,0,0,0> axis_t;

// LeNet-5 Conv2 fixed sizes
constexpr int C_IN   = 1;   // from Pool1
constexpr int H_IN   = 32;
constexpr int W_IN   = 32;
constexpr int C_OUT  = 6;   // Conv2 filters
constexpr int K      = 5;

constexpr int H_CONV = H_IN - K + 1; // 28
constexpr int W_CONV = W_IN - K + 1; // 28

constexpr int S_POOL = 2;            // 2x2 stride 2
constexpr int H_OUT  = H_CONV / S_POOL; // 14
constexpr int W_OUT  = W_CONV / S_POOL; // 14

// Element counts
constexpr int ACT_ELEMS   = C_IN * H_IN    * W_IN;     // 1024
constexpr int W_ELEMS     = C_OUT * C_IN * K * K;      // 150   (6*1*5*5) NOTE: 6*25=150
constexpr int B_ELEMS     = C_OUT;                     // 6
constexpr int CONV_ELEMS  = C_OUT * H_CONV * W_CONV;   // 4704  (6*28*28)
constexpr int OUT_ELEMS   = C_OUT * H_OUT  * W_OUT;    // 1176  (6*14*14)

// Index helpers
static inline int idx_act(int c, int y, int x)          { return (c*H_IN + y)*W_IN + x; }
static inline int idx_w  (int co, int ci, int ky, int kx){ return ((co*C_IN + ci)*K + ky)*K + kx; }
static inline int idx_c  (int co, int y, int x)         { return (co*H_CONV + y)*W_CONV + x; }
static inline int idx_out(int co, int y, int x)         { return (co*H_OUT  + y)*W_OUT  + x; }

// Quantized types
typedef ap_int<8>  q8_t;
typedef ap_int<32> q32_t;

// Pack/unpack 4×int8 in one 32-bit word (little-endian bytes)
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

extern "C" void conv1_preload_axis(
    hls::stream<axis_t> &act_s,
    hls::stream<axis_t> &out_s,
    hls::stream<axis_t> &param_s,
    int load_params,
    int req_shift,
    const q32_t req_m[C_OUT]
) {
#pragma HLS INTERFACE axis      port=act_s
#pragma HLS INTERFACE axis      port=out_s
#pragma HLS INTERFACE axis      port=param_s
#pragma HLS INTERFACE s_axilite port=load_params bundle=control
#pragma HLS INTERFACE s_axilite port=req_shift   bundle=control
#pragma HLS INTERFACE s_axilite port=req_m       bundle=control
#pragma HLS INTERFACE s_axilite port=return      bundle=control

#pragma HLS BIND_OP op=mul impl=DSP
#pragma HLS ALLOCATION operation instances=mul limit=512

    // On-chip persistent parameter storage
    static q8_t  W[W_ELEMS];
    static q32_t B[B_ELEMS];
#pragma HLS BIND_STORAGE variable=W type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=B type=ram_1p impl=bram
    // On-chip buffers
    static q8_t  A[ACT_ELEMS];
    static q8_t  C[CONV_ELEMS]; // conv output (int8 after requant+ReLU)
#pragma HLS BIND_STORAGE variable=A type=ram_1p impl=bram
#pragma HLS BIND_STORAGE variable=C type=ram_1p impl=bram

    // 0) Optional parameter load (once)
    if (load_params) {
        const int w_beats = (W_ELEMS + 3) / 4;
        int wi = 0;
        for (int i = 0; i < w_beats; ++i) {
            axis_t pkt = param_s.read();
            q8_t a0, a1, a2, a3;
            unpack4(pkt.data, a0, a1, a2, a3);
            if (wi < W_ELEMS) W[wi++] = a0;
            if (wi < W_ELEMS) W[wi++] = a1;
            if (wi < W_ELEMS) W[wi++] = a2;
            if (wi < W_ELEMS) W[wi++] = a3;
        }
        for (int i = 0; i < B_ELEMS; ++i) {
            axis_t pkt = param_s.read();
            B[i] = q32_t(ap_int<32>(pkt.data));
        }
    }

    // 1) Read activations for this frame (int8 packed 4 per beat)
    {
        const int a_beats = (ACT_ELEMS + 3) / 4;
        int ai = 0;
        for (int i = 0; i < a_beats; ++i) {
            axis_t pkt = act_s.read();
            q8_t a0, a1, a2, a3;
            unpack4(pkt.data, a0, a1, a2, a3);
            if (ai < ACT_ELEMS) A[ai++] = a0;
            if (ai < ACT_ELEMS) A[ai++] = a1;
            if (ai < ACT_ELEMS) A[ai++] = a2;
            if (ai < ACT_ELEMS) A[ai++] = a3;
        }
    }

    // 2) Conv2 + ReLU + requant (int32 accumulate → int8)
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
                // Fixed-point requantization: s ≈ acc * (req_m[co] / 2^req_shift), round-to-nearest
                q32_t t = acc * req_m[co] + ((req_shift > 0) ? (q32_t(1) << (req_shift - 1)) : q32_t(0));
                q32_t s = (req_shift > 0) ? (t >> req_shift) : t;

                // Saturate to int8 and ReLU
                q8_t q = (s > 127) ? q8_t(127) : (s < -128) ? q8_t(-128) : q8_t(s);
                if (q < q8_t(0)) q = q8_t(0);
                C[idx_c(co, y, x)] = q;
            }
        }
    }

    // 3) MaxPool 2×2 stride 2 and stream out (pack 4 int8 per beat)
    //    Enforce TLAST only on the very final beat.
    const int TOTAL_BEATS = (OUT_ELEMS + 3) / 4;
    int beats_sent = 0;

    q8_t pack_buf[4];
    int   pack_cnt = 0;

    for (int co = 0; co < C_OUT; ++co) {
        for (int y = 0; y < H_OUT; ++y) {
            for (int x = 0; x < W_OUT; ++x) {
                int y0 = y * S_POOL;
                int x0 = x * S_POOL;
                q8_t m = C[idx_c(co, y0,     x0    )];
                q8_t t = C[idx_c(co, y0,     x0 + 1)]; if (t > m) m = t;
                          t = C[idx_c(co, y0 + 1, x0    )]; if (t > m) m = t;
                          t = C[idx_c(co, y0 + 1, x0 + 1)]; if (t > m) m = t;

                pack_buf[pack_cnt++] = m;

                if (pack_cnt == 4) {
                    axis_t pkt;
                    pkt.data = pack4(pack_buf[0], pack_buf[1], pack_buf[2], pack_buf[3]);
                    // Valid 4 bytes
                    pkt.keep = 0xF;
                    pkt.strb = pkt.keep;
                    // Assert TLAST only on the final beat
                    pkt.last = (beats_sent == (TOTAL_BEATS - 1)) ? ap_uint<1>(1) : ap_uint<1>(0);
                    out_s.write(pkt);
                    beats_sent++;
                    pack_cnt = 0;
                }
            }
        }
    }

    // Flush tail if OUT_ELEMS not multiple of 4 (robust handling)
    if (pack_cnt > 0) {
        q8_t a0 = (pack_cnt > 0) ? pack_buf[0] : q8_t(0);
        q8_t a1 = (pack_cnt > 1) ? pack_buf[1] : q8_t(0);
        q8_t a2 = (pack_cnt > 2) ? pack_buf[2] : q8_t(0);
        q8_t a3 = (pack_cnt > 3) ? pack_buf[3] : q8_t(0);

        axis_t pkt;
        pkt.data = pack4(a0, a1, a2, a3);

        // TKEEP must reflect the number of valid bytes in the tail
        ap_uint<4> keep_mask = 0;
        if (pack_cnt >= 1) keep_mask |= 0x1;
        if (pack_cnt >= 2) keep_mask |= 0x2;
        if (pack_cnt >= 3) keep_mask |= 0x4;
        if (pack_cnt >= 4) keep_mask |= 0x8;
        pkt.keep = keep_mask;
        pkt.strb = pkt.keep;

        // Final beat: TLAST must be asserted here
        pkt.last = (beats_sent == (TOTAL_BEATS - 1)) ? ap_uint<1>(1) : ap_uint<1>(0);

        out_s.write(pkt);
        beats_sent++;
    }

    // Optional runtime check (can be removed in synth; useful in sim)
    // Ensures we produced exactly TOTAL_BEATS and TLAST was asserted once.
    // assert(beats_sent == TOTAL_BEATS);
}