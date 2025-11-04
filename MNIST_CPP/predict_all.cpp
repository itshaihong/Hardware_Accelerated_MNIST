#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <algorithm>


static std::vector<char> readFileBytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Failed to open " << path << std::endl; std::exit(1); }
    f.seekg(0, std::ios::end);
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(size);
    if (size > 0) f.read(buf.data(), size);
    return buf;
}

static std::vector<float> read_csv_flat(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("Failed to open CSV: " + path);
    }
    std::vector<float> vals;
    std::string line;
    while (std::getline(f, line)) {
        // Allow comma-separated or single value per line
        size_t start = 0;
        while (start < line.size()) {
            size_t end = line.find(',', start);
            std::string token = (end == std::string::npos) ? line.substr(start) : line.substr(start, end - start);
            if (!token.empty()) {
                vals.push_back(std::stof(token));
            }
            if (end == std::string::npos) break;
            start = end + 1;
        }
    }
    return vals;
}

static std::vector<float> readFloats(const std::string& path) {
    auto bytes = readFileBytes(path);
    if (bytes.size() % sizeof(float) != 0) {
        std::cerr << "File size not multiple of float: " << path << std::endl; std::exit(1);
    }
    size_t n = bytes.size() / sizeof(float);
    std::vector<float> v(n);
    std::memcpy(v.data(), bytes.data(), bytes.size());
    return v;
}

static void read_idx_images(const std::string& path, int& num, int& rows, int& cols, std::vector<float>& images) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Failed to open " << path << std::endl; std::exit(1); }
    uint32_t magic, n, r, c;
    f.read(reinterpret_cast<char*>(&magic), 4);
    f.read(reinterpret_cast<char*>(&n), 4);
    f.read(reinterpret_cast<char*>(&r), 4);
    f.read(reinterpret_cast<char*>(&c), 4);
    auto be32 = [](uint32_t x){ return ((x>>24)&0xff) | ((x>>8)&0xff00) | ((x<<8)&0xff0000) | ((x<<24)&0xff000000); };
    magic = be32(magic); n = be32(n); r = be32(r); c = be32(c);
    if (magic != 2051) { std::cerr << "Invalid images magic" << std::endl; std::exit(1); }
    num = (int)n; rows = (int)r; cols = (int)c;
    std::vector<unsigned char> buf((size_t)num * rows * cols);
    f.read(reinterpret_cast<char*>(buf.data()), buf.size());
    images.resize(buf.size());
    for (size_t i = 0; i < buf.size(); i++) {
        float v = buf[i] / 255.0f;
        images[i] = v;
    }
}

static void read_idx_labels(const std::string& path, int& num, std::vector<unsigned char>& labels) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Failed to open " << path << std::endl; std::exit(1); }
    uint32_t magic, n;
    f.read(reinterpret_cast<char*>(&magic), 4);
    f.read(reinterpret_cast<char*>(&n), 4);
    auto be32 = [](uint32_t x){ return ((x>>24)&0xff) | ((x>>8)&0xff00) | ((x<<8)&0xff0000) | ((x<<24)&0xff000000); };
    magic = be32(magic); n = be32(n);
    if (magic != 2049) { std::cerr << "Invalid labels magic" << std::endl; std::exit(1); }
    num = (int)n;
    labels.resize(num);
    f.read(reinterpret_cast<char*>(labels.data()), labels.size());
}


// ---------- Simple tensor helpers ----------
inline size_t idx3(size_t c, size_t h, size_t w, size_t C, size_t H, size_t W) {
    return c*H*W + h*W + w;
}

inline float relu(float x) { return x > 0.f ? x : 0.f; }

// 2D average pool with kernel=2, stride=2 (no padding)
static void avgpool2x2(const std::vector<float>& in, size_t C, size_t H, size_t W,
                       std::vector<float>& out) {
    size_t Ho = H / 2;
    size_t Wo = W / 2;
    out.assign(C * Ho * Wo, 0.f);
    for (size_t c = 0; c < C; ++c) {
        for (size_t ho = 0; ho < Ho; ++ho) {
            for (size_t wo = 0; wo < Wo; ++wo) {
                size_t h = ho * 2;
                size_t w = wo * 2;
                float s = 0.f;
                s += in[idx3(c, h,   w,   C, H, W)];
                s += in[idx3(c, h,   w+1, C, H, W)];
                s += in[idx3(c, h+1, w,   C, H, W)];
                s += in[idx3(c, h+1, w+1, C, H, W)];
                out[idx3(c, ho, wo, C, Ho, Wo)] = s * 0.25f;
            }
        }
    }
}

// conv2d: input [C_in,H,W], weights [C_out,C_in,Kh,Kw], bias [C_out]
// stride=1, padding as specified (symmetric), activation=ReLU if relu_flag
static void conv2d(const std::vector<float>& in, size_t C_in, size_t H, size_t W,
                   const std::vector<float>& w, const std::vector<float>& b,
                   size_t C_out, size_t Kh, size_t Kw, int pad, bool relu_flag,
                   std::vector<float>& out) {
    size_t Hout = H + 2*pad - Kh + 1;
    size_t Wout = W + 2*pad - Kw + 1;
    out.assign(C_out * Hout * Wout, 0.f);

    for (size_t co = 0; co < C_out; ++co) {
        for (size_t ho = 0; ho < Hout; ++ho) {
            for (size_t wo = 0; wo < Wout; ++wo) {
                float acc = b.empty() ? 0.f : b[co];
                for (size_t ci = 0; ci < C_in; ++ci) {
                    for (size_t kh = 0; kh < Kh; ++kh) {
                        int ih = int(ho + kh) - pad;
                        if (ih < 0 || ih >= (int)H) continue;
                        for (size_t kw = 0; kw < Kw; ++kw) {
                            int iw = int(wo + kw) - pad;
                            if (iw < 0 || iw >= (int)W) continue;
                            float iv = in[idx3(ci, (size_t)ih, (size_t)iw, C_in, H, W)];
                            // weight index: [co, ci, kh, kw]
                            size_t widx = ((co * C_in + ci) * Kh + kh) * Kw + kw;
                            acc += iv * w[widx];
                        }
                    }
                }
                if (relu_flag) acc = relu(acc);
                out[idx3(co, ho, wo, C_out, Hout, Wout)] = acc;
            }
        }
    }
}

// Fully connected: y = W x + b
// W shape [out_features, in_features], x shape [in_features]
static void linear(const std::vector<float>& x,
                   const std::vector<float>& W,
                   const std::vector<float>& b,
                   size_t out_features,
                   std::vector<float>& y) {
    size_t in_features = W.size() / out_features;
    y.assign(out_features, 0.f);
    for (size_t o = 0; o < out_features; ++o) {
        const float* wrow = &W[o * in_features];
        float acc = b.empty() ? 0.f : b[o];
        for (size_t i = 0; i < in_features; ++i) acc += wrow[i] * x[i];
        y[o] = acc;
    }
}



int main() {
    // Paths
    std::string images_path = "../MNIST_python/t10k-images.idx3-ubyte";
    std::string labels_path = "../MNIST_python/t10k-labels.idx1-ubyte";
    std::string kernel_path = "kernel.cl";
    // std::string wpath = "../MNIST_python/weights_csv";

    // // Read weights (float32)
    // auto conv1_w = read_csv_flat(wpath + "/conv1_weight.csv");
    // auto conv1_b = read_csv_flat(wpath + "/conv1_bias.csv");
    // auto conv2_w = read_csv_flat(wpath + "/conv2_weight.csv");
    // auto conv2_b = read_csv_flat(wpath + "/conv2_bias.csv");
    // auto fc1_w   = read_csv_flat(wpath + "/fc1_weight.csv");
    // auto fc1_b   = read_csv_flat(wpath + "/fc1_bias.csv");
    // auto fc2_w   = read_csv_flat(wpath + "/fc2_weight.csv");
    // auto fc2_b   = read_csv_flat(wpath + "/fc2_bias.csv");
    // auto fc3_w   = read_csv_flat(wpath + "/fc3_weight.csv");
    // auto fc3_b   = read_csv_flat(wpath + "/fc3_bias.csv");

    std::string wpath = "../MNIST_python/weights_fp32";

    // Read weights (float32)
    auto conv1_w = readFloats(wpath + "/conv1_weight.bin"); // [6*1*5*5]
    auto conv1_b = readFloats(wpath + "/conv1_bias.bin");   // [6]
    auto conv2_w = readFloats(wpath + "/conv2_weight.bin"); // [16*6*5*5]
    auto conv2_b = readFloats(wpath + "/conv2_bias.bin");   // [16]
    auto fc1_w   = readFloats(wpath + "/fc1_weight.bin");   // [120*400]
    auto fc1_b   = readFloats(wpath + "/fc1_bias.bin");     // [120]
    auto fc2_w   = readFloats(wpath + "/fc2_weight.bin");   // [84*120]
    auto fc2_b   = readFloats(wpath + "/fc2_bias.bin");     // [84]
    auto fc3_w   = readFloats(wpath + "/fc3_weight.bin");   // [10*84]
    auto fc3_b   = readFloats(wpath + "/fc3_bias.bin");     // [10]

    // Read IDX test set
    int num_images, rows, cols;
    std::vector<float> images;
    read_idx_images(images_path, num_images, rows, cols, images);
    int num_labels;
    std::vector<unsigned char> labels;
    read_idx_labels(labels_path, num_labels, labels);
    if (num_images != num_labels || rows != 28 || cols != 28) {
        std::cerr << "Dataset mismatch" << std::endl; return 1;
    }

    // Normalize same as training: (x - 0.1307) / 0.3081
    const float mean = 0.1307f, stdv = 0.3081f;
    for (size_t i = 0; i < images.size(); i++) {
        images[i] = (images[i] - mean) / stdv;

    }

    // Verify expected shapes to catch mismatches early
    if (conv1_w.size() != 6u*1u*5u*5u) { std::cerr << "conv1_w size mismatch\n"; return 1; }
    if (conv2_w.size() != 16u*6u*5u*5u) { std::cerr << "conv2_w size mismatch\n"; return 1; }
    if (fc1_w.size() != 120u*400u) { std::cerr << "fc1_w size mismatch (expect 120x400)\n"; return 1; }
    if (fc2_w.size() != 84u*120u) { std::cerr << "fc2_w size mismatch\n"; return 1; }
    if (fc3_w.size() != 10u*84u) { std::cerr << "fc3_w size mismatch\n"; return 1; }

    size_t correct = 0;
    auto t0 = std::chrono::high_resolution_clock::now();

    // Buffers reused across images to reduce allocations
    std::vector<float> x0(1*28*28);
    std::vector<float> c1, p1, c2, p2, flat, y1, y2, logits;

    for (int i = 0; i < num_images; ++i) {
        // Load one image into x0 [C=1,H=28,W=28]
        const float* src = &images[(size_t)i * 28 * 28];
        std::memcpy(x0.data(), src, 28u * 28u * sizeof(float));

        // conv1: [1,28,28] + 6 filters 5x5, pad=2, ReLU -> [6,28,28]
        conv2d(x0, /*C_in=*/1, /*H=*/28, /*W=*/28,
               conv1_w, conv1_b,
               /*C_out=*/6, /*Kh=*/5, /*Kw=*/5, /*pad=*/2, /*relu=*/true,
               c1);

        // maxpool2x2 2x2 -> [6,14,14]
        maxpool2x2(c1, /*C=*/6, /*H=*/28, /*W=*/28, p1);

        // conv2: [6,14,14] + 16 filters 5x5, pad=0, ReLU -> [16,10,10]
        conv2d(p1, /*C_in=*/6, /*H=*/14, /*W=*/14,
               conv2_w, conv2_b,
               /*C_out=*/16, /*Kh=*/5, /*Kw=*/5, /*pad=*/0, /*relu=*/true,
               c2);

        // maxpool2x2 2x2 -> [16,5,5]
        maxpool2x2(c2, /*C=*/16, /*H=*/10, /*W=*/10, p2);

        // flatten [16*5*5=400]
        flat.assign(16u * 5u * 5u, 0.f);
        for (size_t c = 0; c < 16; ++c)
            for (size_t h = 0; h < 5; ++h)
                for (size_t w = 0; w < 5; ++w)
                    flat[c*25 + h*5 + w] = p2[idx3(c, h, w, 16, 5, 5)];

        // fc1: 400 -> 120, ReLU
        linear(flat, fc1_w, fc1_b, /*out_features=*/120, y1);
        for (auto& v : y1) v = relu(v);

        // fc2: 120 -> 84, ReLU
        linear(y1, fc2_w, fc2_b, /*out_features=*/84, y2);
        for (auto& v : y2) v = relu(v);

        // fc3: 84 -> 10 (logits)
        linear(y2, fc3_w, fc3_b, /*out_features=*/10, logits);

        // argmax
        int pred = int(std::max_element(logits.begin(), logits.end()) - logits.begin());
        if (pred == int(labels[i])) ++correct;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    double fps = double(num_images) / secs;
    double avg_ms = (secs * 1000.0) / double(num_images);

    std::cout << "\n=== Results (Pure C++ CPU) ===\n";
    std::cout << "Accuracy: " << (100.0 * correct / num_images) << "% (" << correct << "/" << num_images << ")\n";
    std::cout << "Average inference time: " << avg_ms << " ms\n";
    std::cout << "Throughput: " << fps << " FPS\n";

    return 0;
}