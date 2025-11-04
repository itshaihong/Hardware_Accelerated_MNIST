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
    const qint8*  in_q,    // AXI m_axi
    const qint8*  w_q,     // AXI m_axi
    const qint16* b_q,     // AXI m_axi
    qint8*        out_q,   // AXI m_axi
    int Cin,
    int H,
    int W,
    int Cout,
    int pad,               // typically 0 or 2 for LeNet
    int pool,              // 0: none, 1: maxpool 2x2
    float scale_S,         // post-accum scaling (e.g., for int8 quant)
    int shift              // arithmetic right shift after scaling (optional)
);
}