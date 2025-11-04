// OpenCL kernels for LeNet-5 inference (float32)

__kernel void conv2d_relu(
    __global const float* input,
    __global const float* weight,
    __global const float* bias,
    int in_c, int out_c,
    int in_h, int in_w,
    int k, int stride, int pad,
    int out_h, int out_w,
    __global float* output)
{
    int gid = get_global_id(0);
    int spatial = out_h * out_w;
    int oc = gid / spatial;
    int rem = gid % spatial;
    int oh = rem / out_w;
    int ow = rem % out_w;
    if (oc >= out_c) return;

    float acc = bias[oc];
    for (int ic = 0; ic < in_c; ic++) {
        for (int kh = 0; kh < k; kh++) {
            for (int kw = 0; kw < k; kw++) {
                int ih = oh * stride + kh - pad;
                int iw = ow * stride + kw - pad;
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    int in_idx = ic * in_h * in_w + ih * in_w + iw;
                    int w_idx = oc * in_c * k * k + ic * k * k + kh * k + kw;
                    acc += input[in_idx] * weight[w_idx];
                }
            }
        }
    }
    // ReLU
    if (acc < 0.0f) acc = 0.0f;
    output[oc * out_h * out_w + oh * out_w + ow] = acc;
}

__kernel void maxpool2x2(
    __global const float* input,
    int channels, int in_h, int in_w,
    __global float* output)
{
    int out_h = in_h / 2;
    int out_w = in_w / 2;
    int gid = get_global_id(0);
    int spatial = out_h * out_w;
    int c = gid / spatial;
    int rem = gid % spatial;
    int oh = rem / out_w;
    int ow = rem % out_w;
    if (c >= channels) return;

    float m = -3.4e38f;
    for (int ph = 0; ph < 2; ph++) {
        for (int pw = 0; pw < 2; pw++) {
            int ih = oh * 2 + ph;
            int iw = ow * 2 + pw;
            float v = input[c * in_h * in_w + ih * in_w + iw];
            if (v > m) m = v;
        }
    }
    output[c * out_h * out_w + oh * out_w + ow] = m;
}

__kernel void fc_relu(
    __global const float* input,
    __global const float* weight, // [out, in]
    __global const float* bias,   // [out]
    int in_size, int out_size,
    __global float* output)
{
    int o = get_global_id(0);
    if (o >= out_size) return;
    float acc = bias[o];
    int w_base = o * in_size;
    for (int i = 0; i < in_size; i++) {
        acc += input[i] * weight[w_base + i];
    }
    if (acc < 0.0f) acc = 0.0f;
    output[o] = acc;
}

__kernel void fc_norelu(
    __global const float* input,
    __global const float* weight, // [out, in]
    __global const float* bias,   // [out]
    int in_size, int out_size,
    __global float* output)
{
    int o = get_global_id(0);
    if (o >= out_size) return;
    float acc = bias[o];
    int w_base = o * in_size;
    for (int i = 0; i < in_size; i++) {
        acc += input[i] * weight[w_base + i];
    }
    output[o] = acc;
}