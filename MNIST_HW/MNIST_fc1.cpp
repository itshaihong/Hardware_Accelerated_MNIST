#include "hls_stream.h"
#include "ap_int.h"
#include "ap_axi_sdata.h"

#define ROWS_A 120
#define COLS_A 400
#define COLS_B 1   // for clarity

// AXI stream data type: 32-bit data + control signals
typedef ap_axis<32,0,0,0> AXIS;

void fc1(hls::stream<AXIS>& S_AXIS, hls::stream<AXIS>& M_AXIS){
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis port=S_AXIS
#pragma HLS INTERFACE axis port=M_AXIS

    // Local buffers for inputs
    int A[ROWS_A][COLS_A];
    int B[COLS_A];
    int RES[ROWS_A];

#pragma HLS ARRAY_PARTITION variable=A dim=2 complete   // optimize parallelism on inner dimension
#pragma HLS ARRAY_PARTITION variable=B complete


    AXIS axis_in, axis_out;

    // -------------------- Read Matrix A (64x8 = 512 words) --------------------
    for(int i = 0; i < ROWS_A; i++){
        for(int j = 0; j < COLS_A; j++){
#pragma HLS PIPELINE II=1

            axis_in = S_AXIS.read();
            A[i][j] = axis_in.data.to_int();
        }
    }

    // -------------------- Read Matrix B (8x1 = 8 words) --------------------
    for(int j = 0; j < COLS_A; j++){
#pragma HLS PIPELINE II=1

        axis_in = S_AXIS.read();
        B[j] = axis_in.data.to_int();
    }

    // -------------------- Compute RES = A*B/256 --------------------
    for(int i = 0; i < ROWS_A; i++){
        int sum = 0;
        for(int j = 0; j < COLS_A; j++){
#pragma PIPELINE II=1 

            sum += A[i][j] * B[j];
        }
        RES[i] = sum >> 8;   // divide by 256 using shift
    }

    // -------------------- Write output RES (64 words) --------------------
    for(int i = 0; i < ROWS_A; i++){
#pragma HLS PIPELINE II=1

        axis_out.data = RES[i];
        axis_out.keep = 0xF;
        axis_out.strb = 0xF;
        axis_out.last = (i == ROWS_A-1) ? 1 : 0;  // mark last word
        M_AXIS.write(axis_out);
    }
}