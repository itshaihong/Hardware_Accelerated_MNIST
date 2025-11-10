#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <cmath>
#include <cstdint>
#include <cstring>

typedef ap_axis<32,0,0,0> AXIS; // 32-bit TDATA stream

// Fixed LeNet-5 first block sizes
#define C_IN  1
#define H_IN  32
#define W_IN  32
#define C1_OUT  6
#define K       5
#define H1      28 // 28
#define W1      28 // 28
#define S_POOL  2           // 2x2, stride 2
#define H1P     14  // 14
#define W1P     14  // 14
#define IMG_ELEMS  1024            
#define W_ELEMS    150         
#define B_ELEMS    6                      
#define IN_TOTAL   1180
#define CONV_ELEMS  4704           
#define OUT_ELEMS   1176           // 1176

// Index helpers
static inline int idx_img(int c, int y, int x)                 { return (c*H_IN + y)*W_IN + x; }
static inline int idx_w  (int co, int ci, int ky, int kx)      { return ((co*C_IN + ci)*K + ky)*K + kx; }
static inline int idx_c1 (int co, int y, int x)                { return (co*H1 + y)*W1 + x; }
static inline int idx_p1 (int co, int y, int x)                { return (co*H1P + y)*W1P + x; }

void conv1_hardcode(hls::stream<AXIS>& in_s,
                                           hls::stream<AXIS>& out_s)
{
#pragma HLS INTERFACE axis      port=in_s   
#pragma HLS INTERFACE axis      port=out_s  
#pragma HLS INTERFACE ap_ctrl_none port=return

// DSP binding for arithmetic (reduce LUT/CARRY usage)
#pragma HLS BIND_OP op=mul impl=DSP
#pragma HLS ALLOCATION operation instances=mul limit=512

    // Local storage (BRAM/URAM as needed)
    int img[IMG_ELEMS];
    int w  [W_ELEMS];
    int b  [B_ELEMS];
    int RES[OUT_ELEMS];

    static int conv_buf[CONV_ELEMS];
#pragma HLS BIND_STORAGE variable=conv_buf type=ram_1p impl=bram

    AXIS axis_in, axis_out;

    // 1) Read input payload: IMG -> W -> B
    for (int i = 0; i < IMG_ELEMS; ++i) {
#pragma HLS PIPELINE II=1
        axis_in = in_s.read();
        img[i] = axis_in.data;
    }
    for (int i = 0; i < W_ELEMS; ++i) {
#pragma HLS PIPELINE II=1
        axis_in = in_s.read();
        w[i] = axis_in.data;
    }
    for (int i = 0; i < B_ELEMS; ++i) {
#pragma HLS PIPELINE II=1
        axis_in = in_s.read();
        b[i] = axis_in.data;
    }
    // We rely on fixed sizes; TLAST on input is not required internally.

    // 2) Conv1 + ReLU (valid)
    for (int y = 0; y < H1; ++y) {
        for (int x = 0; x < W1; ++x) {
#pragma HLS PIPELINE II=1
            for (int co = 0; co < C1_OUT; ++co) {
                int acc = 0;
                for (int ci = 0; ci < C_IN; ++ci) {
                    for (int ky = 0; ky < K; ++ky) {
                        for (int kx = 0; kx < K; ++kx) {
                            int vin = img[idx_img(ci, y + ky, x + kx)];
                            int wgt = w[idx_w(co, ci, ky, kx)];
                            acc += (vin * wgt);
                        }
                    }
                }
                acc = acc >> 8;
                acc = acc + b[co];
                if (acc < (int)0) acc = (int)0; // ReLU
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
                int m = conv_buf[idx_c1(co, y0,     x0    )];
                int t = conv_buf[idx_c1(co, y0,     x0 + 1)];
                if (t > m) m = t;
                t =        conv_buf[idx_c1(co, y0 + 1, x0    )];
                if (t > m) m = t;
                t =        conv_buf[idx_c1(co, y0 + 1, x0 + 1)];
                if (t > m) m = t;
                RES[out_index] = m;
                out_index++;
            }
        }
    }

        for(int i = 0; i < OUT_ELEMS; i++){
#pragma HLS PIPELINE II=1

        axis_out.data = RES[i];
        axis_out.keep = 0xF;
        axis_out.strb = 0xF;
        axis_out.last = (i == OUT_ELEMS-1) ? 1 : 0;  // mark last word
        out_s.write(axis_out);
    }
}
