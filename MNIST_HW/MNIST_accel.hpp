#pragma once
#include <ap_int.h>
#include <stdint.h>

typedef ap_int<8>  qint8;
typedef ap_int<16> qint16;

// Pooling control
enum PoolType { POOL_NONE = 0, POOL_MAX2x2 = 1 };

// Top-level HLS accelerator: Conv5x5 (stride=1, padding=pad) + ReLU + optional MaxPool 2x2
// Layouts (row-major):
// - in_q:  [Cin][H][W]
// - w_q:   [Cout][Cin][5][5]
// - b_q:   [Cout] (bias in accumulator domain)
// - out_q: If pool=0: [Cout][H+2*pad-4][W+2*pad-4]
//          If pool=1: [Cout][(H+2*pad-4)/2][(W+2*pad-4)/2]
extern "C" {
void cnn_accel(
    const qint8*  in_q,    // [Cin*H*W]
    const qint8*  w_q,     // [Cout*Cin*5*5]
    const qint16* b_q,     // [Cout]
    qint8*        out_q,   // [Cout*Hout*Wout]
    int Cin,
    int H,
    int W,
    int Cout,
    int pad,
    int pool,
    float scale_S,
    int shift
);
}