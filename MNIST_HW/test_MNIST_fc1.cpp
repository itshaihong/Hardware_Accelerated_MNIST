#include <stdio.h>
#include "hls_stream.h"
#include "ap_axi_sdata.h"

typedef ap_axis<32,0,0,0> AXIS;

// Coprocessor function declaration
void fc1(hls::stream<AXIS>& S_AXIS, hls::stream<AXIS>& M_AXIS);

#define ROWS_A 120
#define COLS_A 400
#define COLS_B 1

/************************** Global buffers *****************************/
// Input matrices
int A[ROWS_A][COLS_A];
int B[COLS_A];

// Expected and actual result vectors
int RES_expected[ROWS_A];
int RES_hw[ROWS_A];

int main() {
    int i, j;
    int success = 1;

    hls::stream<AXIS> S_AXIS;
    hls::stream<AXIS> M_AXIS;
    AXIS axis_in, axis_out;

    /******************** Initialize input data *********************/
    // Example: A[i][j] = i+j , B[j] = j
    for (i = 0; i < ROWS_A; i++) {
        for (j = 0; j < COLS_A; j++) {
            A[i][j] = (i + j) & 0xFF;  // keep values small
        }
    }
    for (j = 0; j < COLS_A; j++) {
        B[j] = (j+1);   // 1,2,3,4,5,6,7,8
    }

    /******************** Generate expected result *********************/
    for (i = 0; i < ROWS_A; i++) {
        int sum = 0;
        for (j = 0; j < COLS_A; j++) {
            sum += A[i][j] * B[j];
        }
        RES_expected[i] = sum >> 8;  // divide by 256
    }

    /******************** Send input to coprocessor *********************/
    // First send A (64x8 = 512 words)
    for (i = 0; i < ROWS_A; i++) {
        for (j = 0; j < COLS_A; j++) {
            axis_in.data = A[i][j];
            axis_in.keep = 0xF;
            axis_in.strb = 0xF;
            axis_in.last = 0;
            S_AXIS.write(axis_in);
        }
    }

    // Then send B (8x1 = 8 words)
    for (j = 0; j < COLS_A; j++) {
        axis_in.data = B[j];
        axis_in.keep = 0xF;
        axis_in.strb = 0xF;
        axis_in.last = (j == COLS_A-1) ? 1 : 0;  // last word of input stream
        S_AXIS.write(axis_in);
    }

    /******************** Call hardware function *********************/
    fc1(S_AXIS, M_AXIS);

    /******************** Receive output from coprocessor *********************/
    for (i = 0; i < ROWS_A; i++) {
        axis_out = M_AXIS.read();
        RES_hw[i] = axis_out.data.to_int();
    }

    /******************** Compare results *********************/
    printf("Comparing results ...\n");
    for (i = 0; i < ROWS_A; i++) {
        if (RES_hw[i] != RES_expected[i]) {
            printf("Mismatch at row %d: HW=%d, SW=%d\n", i, RES_hw[i], RES_expected[i]);
            success = 0;
        }
    }

    if (success) {
        printf("Test Success!\n");
    } else {
        printf("Test Failed!\n");
    }

    return success ? 0 : 1;
}
