#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <cmath>
#include <cstdint>
#include <cstring>

using data_t = float;
typedef ap_axiu<32,0,0,0> axis_t; // 32-bit TDATA stream

// Fixed LeNet-5 first block sizes
constexpr int C_IN = 1;
constexpr int H_IN = 28;
constexpr int W_IN = 28;

constexpr int C1_OUT = 6;
constexpr int K      = 5;
constexpr int H1     = H_IN - K + 1; // 24
constexpr int W1     = W_IN - K + 1; // 24

constexpr int S_POOL = 2;            // 2x2, stride 2
constexpr int H1P    = H1 / S_POOL;  // 12
constexpr int W1P    = W1 / S_POOL;  // 12

// Element counts
constexpr int IMG_ELEMS = C_IN * H_IN * W_IN;            // 784
constexpr int W_ELEMS   = C1_OUT * C_IN * K * K;         // 150
constexpr int B_ELEMS   = C1_OUT;                        // 6
constexpr int IN_TOTAL  = IMG_ELEMS + W_ELEMS + B_ELEMS; // 940

constexpr int CONV_ELEMS = C1_OUT * H1  * W1;            // 3456
constexpr int OUT_ELEMS  = C1_OUT * H1P * W1P;           // 864

// Index helpers
static inline int idx_img(int c, int y, int x)                 { return (c*H_IN + y)*W_IN + x; }
static inline int idx_w  (int co, int ci, int ky, int kx)      { return ((co*C_IN + ci)*K + ky)*K + kx; }
static inline int idx_c1 (int co, int y, int x)                { return (co*H1 + y)*W1 + x; }
static inline int idx_p1 (int co, int y, int x)                { return (co*H1P + y)*W1P + x; }

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
                                           hls::stream<axis_t>& out_s)
{
#pragma HLS INTERFACE axis      port=in_s   
#pragma HLS INTERFACE axis      port=out_s  
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local storage (BRAM/URAM as needed)
    static data_t img[IMG_ELEMS];
    static data_t w  [W_ELEMS];
    static data_t b  [B_ELEMS];

    static data_t conv_buf[CONV_ELEMS];
#pragma HLS BIND_STORAGE variable=conv_buf type=ram_1p impl=bram

    // 1) Read input payload: IMG -> W -> B
    for (int i = 0; i < IMG_ELEMS; ++i) {
#pragma HLS PIPELINE II=1
        axis_t pkt = in_s.read();
        img[i] = u2f(pkt.data);
    }
    for (int i = 0; i < W_ELEMS; ++i) {
#pragma HLS PIPELINE II=1
        axis_t pkt = in_s.read();
        w[i] = u2f(pkt.data);
    }
    for (int i = 0; i < B_ELEMS; ++i) {
#pragma HLS PIPELINE II=1
        axis_t pkt = in_s.read();
        b[i] = u2f(pkt.data);
    }
    // We rely on fixed sizes; TLAST on input is not required internally.

    // 2) Conv1 + ReLU (valid)
    for (int y = 0; y < H1; ++y) {
        for (int x = 0; x < W1; ++x) {
#pragma HLS PIPELINE II=1
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
                if (acc < (data_t)0) acc = (data_t)0; // ReLU
                conv_buf[idx_c1(co, y, x)] = acc;
            }
        }
    }

    // 3) MaxPool (2x2, stride 2) and stream out
    int out_index = 0;
    for (int co = 0; co < C1_OUT; ++co) {
        for (int y = 0; y < H1P; ++y) {
            for (int x = 0; x < W1P; ++x) {
        #pragma HLS PIPELINE II=1
                int y0 = y * S_POOL;
                int x0 = x * S_POOL;
                data_t m = conv_buf[idx_c1(co, y0,     x0    )];
                data_t t = conv_buf[idx_c1(co, y0,     x0 + 1)];
                if (t > m) m = t;
                t =        conv_buf[idx_c1(co, y0 + 1, x0    )];
                if (t > m) m = t;
                t =        conv_buf[idx_c1(co, y0 + 1, x0 + 1)];
                if (t > m) m = t;

                axis_t pkt{};
                pkt.data = f2u(m);
                pkt.keep = -1; // all 4 bytes valid
                pkt.last = (out_index == OUT_ELEMS - 1) ? 1 : 0;
                out_s.write(pkt);
                ++out_index;
            }
        }
    }
}