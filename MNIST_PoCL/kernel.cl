// Later, consider fusing pool into the conv kernel for further speed if profiling shows memory bandwidth is the bottleneck.
__kernel void conv2d_bias_relu(
    __global const float* x,          // [C_in, H, W]
    __global const float* w,          // [C_out, C_in, KH, KW]
    __global const float* b,          // [C_out]
    __global float* y,                // [C_out, H_out, W_out]
    int C_in, int H, int W,
    int C_out, int KH, int KW,
    int pad_h, int pad_w)
{
    int ow = get_global_id(0);
    int oh = get_global_id(1);
    int oc = get_global_id(2);
    int H_out = H + 2*pad_h - KH + 1;
    int W_out = W + 2*pad_w - KW + 1;
    if (ow >= W_out || oh >= H_out || oc >= C_out) return;

    float acc = b[oc];
    for (int ic = 0; ic < C_in; ++ic) {
        for (int kh = 0; kh < KH; ++kh) {
            int ih = oh + kh - pad_h;
            if ((unsigned)ih >= (unsigned)H) continue;
            for (int kw = 0; kw < KW; ++kw) {
                int iw = ow + kw - pad_w;
                if ((unsigned)iw >= (unsigned)W) continue;
                // index helpers for NCHW (flattened)
                int xidx = (ic*H + ih)*W + iw;
                int widx = (((oc*C_in) + ic)*KH + kh)*KW + kw;
                acc += x[xidx] * w[widx];
            }
        }
    }
    // ReLU
    if (acc < 0.0f) acc = 0.0f;
    int yidx = (oc*H_out + oh)*W_out + ow;
    y[yidx] = acc;
}

__kernel void maxpool2x2(
    __global const float* x,  // [C, H, W]
    __global float* y,        // [C, H/2, W/2]
    int C, int H, int W)
{
    int ow = get_global_id(0);
    int oh = get_global_id(1);
    int c  = get_global_id(2);
    int H_out = H / 2, W_out = W / 2;
    if (ow >= W_out || oh >= H_out || c >= C) return;

    float m = -FLT_MAX;
    for (int dh = 0; dh < 2; ++dh)
        for (int dw = 0; dw < 2; ++dw) {
            int ih = oh*2 + dh;
            int iw = ow*2 + dw;
            float v = x[(c*H + ih)*W + iw];
            m = fmax(m, v);
        }
    y[(c*H_out + oh)*W_out + ow] = m;
}

__kernel void linear_gemv(
    __global const float* W,  // [O, I]
    __global const float* b,  // [O]
    __global const float* x,  // [I]
    __global float* y,        // [O]
    int O, int I)
{
    int o = get_global_id(0);
    if (o >= O) return;
    float acc = b[o];
    int row = o * I;
    for (int i = 0; i < I; ++i) {
        acc += W[row + i] * x[i];
    }
    y[o] = acc; // apply ReLU here if desired
}