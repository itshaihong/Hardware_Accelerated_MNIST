// predict_single.cpp
// C++ equivalent of the Python single-image MNIST inference using a LeNet-5 CNN.
// This version performs CPU-only inference without PyTorch.
// It expects weights exported to CSV (flattened) for LeNet-5 as described earlier:
//   conv1_weight.csv (shape: 6*1*5*5 values, flattened)
//   conv1_bias.csv   (shape: 6)
//   conv2_weight.csv (shape: 16*6*5*5)
//   conv2_bias.csv   (shape: 16)
//   fc1_weight.csv   (shape: 120*400)
//   fc1_bias.csv     (shape: 120)
//   fc2_weight.csv   (shape: 84*120)
//   fc2_bias.csv     (shape: 84)
//   fc3_weight.csv   (shape: 10*84)
//   fc3_bias.csv     (shape: 10)
//
// Build requirements: OpenCV (for image I/O)
// Compile example (Ubuntu):
//   sudo apt update
//   sudo apt install -y build-essential pkg-config libopencv-dev
//   g++ -O3 -std=c++17 predict_single.cpp -o predict_single `pkg-config --cflags --libs opencv4`
//
// Run example:
//   ./predict_single --image pixil_8.png --weights_dir weights_csv --topk 3 --invert auto --save_preprocessed preproc.png
//
// Notes:
// - This program uses flattened CSV weights (one value per line). If you saved non-flattened CSVs,
//   adjust the CSV loader accordingly.
// - Normalization is identical to training: (x - 0.1307) / 0.3081 for inputs scaled to [0,1].
// - Convolution layout: out_ch × in_ch × kH × kW (PyTorch default). Linear: out × in.

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

static constexpr float MEAN = 0.1307f;
static constexpr float STD = 0.3081f;

struct Args {
    std::string image = "..\\MNIST_python\\test_3.png";
    std::string weights_dir = "..\\MNIST_python\\weights_csv";
    std::string invert = "auto"; // "auto", "yes", "no"
    int topk = 3;
    std::optional<std::string> save_preprocessed;
};

bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* err) -> std::string {
            if (i + 1 >= argc) { std::cerr << err << "\n"; exit(1); }
            return std::string(argv[++i]);
        };
        if (a == "--image") {
            args.image = next("Missing value for --image");
        } else if (a == "--weights_dir") {
            args.weights_dir = next("Missing value for --weights_dir");
        } else if (a == "--invert") {
            args.invert = next("Missing value for --invert");
            if (args.invert != "auto" && args.invert != "yes" && args.invert != "no") {
                std::cerr << "Invalid --invert. Use auto|yes|no.\n";
                return false;
            }
        } else if (a == "--topk") {
            args.topk = std::stoi(next("Missing value for --topk"));
        } else if (a == "--save_preprocessed") {
            args.save_preprocessed = next("Missing value for --save_preprocessed");
        } else {
            std::cerr << "Unknown argument: " << a << "\n";
            return false;
        }
    }
    if (args.image.empty()) {
        std::cerr << "Usage: " << argv[0] << " --image <path> [--weights_dir dir] [--invert auto|yes|no] [--topk K] [--save_preprocessed path]\n";
        return false;
    }
    return true;
}

std::vector<float> read_csv_flat(const std::string& path) {
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

struct Weights {
    // Conv1
    std::vector<float> conv1_w; // [6,1,5,5] flattened
    std::vector<float> conv1_b; // [6]
    // Conv2
    std::vector<float> conv2_w; // [16,6,5,5]
    std::vector<float> conv2_b; // [16]
    // FCs
    std::vector<float> fc1_w; // [120,400]
    std::vector<float> fc1_b; // [120]
    std::vector<float> fc2_w; // [84,120]
    std::vector<float> fc2_b; // [84]
    std::vector<float> fc3_w; // [10,84]
    std::vector<float> fc3_b; // [10]
};

Weights load_weights(const std::string& dir) {
    Weights w;
    auto path = [&](const std::string& name) { return dir + "/" + name; };

    w.conv1_w = read_csv_flat(path("conv1_weight.csv"));
    w.conv1_b = read_csv_flat(path("conv1_bias.csv"));
    w.conv2_w = read_csv_flat(path("conv2_weight.csv"));
    w.conv2_b = read_csv_flat(path("conv2_bias.csv"));
    w.fc1_w = read_csv_flat(path("fc1_weight.csv"));
    w.fc1_b = read_csv_flat(path("fc1_bias.csv"));
    w.fc2_w = read_csv_flat(path("fc2_weight.csv"));
    w.fc2_b = read_csv_flat(path("fc2_bias.csv"));
    w.fc3_w = read_csv_flat(path("fc3_weight.csv"));
    w.fc3_b = read_csv_flat(path("fc3_bias.csv"));

    // Basic shape checks
    if (w.conv1_w.size() != 6 * 1 * 5 * 5 || w.conv1_b.size() != 6) throw std::runtime_error("conv1 shapes mismatch");
    if (w.conv2_w.size() != 16 * 6 * 5 * 5 || w.conv2_b.size() != 16) throw std::runtime_error("conv2 shapes mismatch");
    if (w.fc1_w.size() != 120 * 400 || w.fc1_b.size() != 120) throw std::runtime_error("fc1 shapes mismatch");
    if (w.fc2_w.size() != 84 * 120 || w.fc2_b.size() != 84) throw std::runtime_error("fc2 shapes mismatch");
    if (w.fc3_w.size() != 10 * 84 || w.fc3_b.size() != 10) throw std::runtime_error("fc3 shapes mismatch");

    return w;
}

// Tensor helper accessors for flattened arrays
inline size_t idx4(size_t c, size_t ic, size_t kh, size_t kw, size_t IC, size_t KH, size_t KW) {
    return (((c * IC + ic) * KH + kh) * KW + kw);
}
inline size_t idx3(size_t c, size_t h, size_t w, size_t H, size_t W) {
    return ((c * H + h) * W + w);
}

// ReLU in-place
void relu(std::vector<float>& x) {
    for (auto& v : x) if (v < 0.0f) v = 0.0f;
}

// MaxPool2d 2x2 stride 2: input [C,H,W] -> output [C,H/2,W/2]
std::vector<float> maxpool2x2(const std::vector<float>& x, size_t C, size_t H, size_t W) {
    size_t Ho = H / 2, Wo = W / 2;
    std::vector<float> y(C * Ho * Wo, 0.0f);
    for (size_t c = 0; c < C; ++c) {
        for (size_t i = 0; i < Ho; ++i) {
            for (size_t j = 0; j < Wo; ++j) {
                float m = -std::numeric_limits<float>::infinity();
                for (size_t di = 0; di < 2; ++di) {
                    for (size_t dj = 0; dj < 2; ++dj) {
                        size_t h = i * 2 + di;
                        size_t w = j * 2 + dj;
                        m = std::max(m, x[idx3(c, h, w, H, W)]);
                    }
                }
                y[idx3(c, i, j, Ho, Wo)] = m;
            }
        }
    }
    return y;
}

// Conv2d: input [IC,H,W], weights [OC,IC,KH,KW], bias [OC], padding P (same padding on H,W), stride 1
std::vector<float> conv2d(const std::vector<float>& x, size_t IC, size_t H, size_t W,
                          const std::vector<float>& w, const std::vector<float>& b,
                          size_t OC, size_t KH, size_t KW, int P) {
    size_t Ho = H + 2 * P - KH + 1;
    size_t Wo = W + 2 * P - KW + 1;
    std::vector<float> y(OC * Ho * Wo, 0.0f);

    for (size_t oc = 0; oc < OC; ++oc) {
        for (size_t oh = 0; oh < Ho; ++oh) {
            for (size_t ow = 0; ow < Wo; ++ow) {
                float acc = b[oc];
                for (size_t ic = 0; ic < IC; ++ic) {
                    for (size_t kh = 0; kh < KH; ++kh) {
                        int ih = static_cast<int>(oh) + static_cast<int>(kh) - P;
                        if (ih < 0 || ih >= static_cast<int>(H)) continue;
                        for (size_t kw = 0; kw < KW; ++kw) {
                            int iw = static_cast<int>(ow) + static_cast<int>(kw) - P;
                            if (iw < 0 || iw >= static_cast<int>(W)) continue;
                            float xv = x[idx3(ic, (size_t)ih, (size_t)iw, H, W)];
                            float wv = w[idx4(oc, ic, kh, kw, IC, KH, KW)];
                            acc += xv * wv;
                        }
                    }
                }
                y[idx3(oc, oh, ow, Ho, Wo)] = acc;
            }
        }
    }
    return y;
}

// Linear: y = W x + b; W [O,I], x [I], b [O]
std::vector<float> linear(const std::vector<float>& W, const std::vector<float>& b, size_t O, size_t I, const std::vector<float>& x) {
    std::vector<float> y(O, 0.0f);
    for (size_t o = 0; o < O; ++o) {
        float acc = b[o];
        const size_t row_start = o * I;
        for (size_t i = 0; i < I; ++i) {
            acc += W[row_start + i] * x[i];
        }
        y[o] = acc;
    }
    return y;
}

std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> y(logits.size());
    float maxv = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        y[i] = std::exp(logits[i] - maxv);
        sum += y[i];
    }
    for (size_t i = 0; i < logits.size(); ++i) {
        y[i] = static_cast<float>(y[i] / sum);
    }
    return y;
}

// Load, preprocess image to [1,28,28], normalize, and optionally save preprocessed PNG
std::vector<float> load_and_preprocess(const Args& args) {
    // Load grayscale image
    cv::Mat img = cv::imread(args.image, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        throw std::runtime_error("Failed to read image: " + args.image);
    }

    // Resize to 28x28 with nearest-neighbor to keep digit edges crisp
    if (img.cols != 28 || img.rows != 28) {
        cv::resize(img, img, cv::Size(28, 28), 0, 0, cv::INTER_NEAREST);
    }

    // Convert to float [0,1]
    img.convertTo(img, CV_32F, 1.0 / 255.0);

    // Decide inversion
    bool do_invert = false;
    if (args.invert == "yes") {
        do_invert = true;
    } else if (args.invert == "no") {
        do_invert = false;
    } else { // auto
        cv::Scalar m = cv::mean(img);
        // If background is bright (mean > 0.5), invert to get white digit on black background
        if (m[0] > 0.5) do_invert = true;
    }

    if (do_invert) {
        // Invert intensities: x <- 1 - x
        img = 1.0f - img;
    }

    // Optionally save the preprocessed image (post-resize, post-inversion), as uint8 0..255 PNG
    if (args.save_preprocessed.has_value()) {
        cv::Mat to_save;
        img.convertTo(to_save, CV_8U, 255.0);
        if (!cv::imwrite(args.save_preprocessed.value(), to_save)) {
            std::cerr << "Warning: failed to save preprocessed image to "
                      << args.save_preprocessed.value() << "\n";
        }
    }

    // Normalize using training parameters: (x - MEAN) / STD
    img -= MEAN;
    img /= STD;

    // Ensure contiguous memory
    if (!img.isContinuous()) {
        img = img.clone();
    }

    // Copy to vector<float> in row-major order
    std::vector<float> out(28 * 28);
    std::memcpy(out.data(), img.ptr<float>(0), out.size() * sizeof(float));
    return out;
}


int main(int argc, char** argv) {
    try {
        // Parse CLI
        Args args;
        if (!parse_args(argc, argv, args)) {
            return 1;
        }

        // Load weights (CSV, flattened)
        Weights W = load_weights(args.weights_dir);

        // Load and preprocess image -> vector<float> of size 28*28, normalized
        std::vector<float> x = load_and_preprocess(args);
        if (x.size() != 28 * 28) {
            std::cerr << "Preprocessed input has wrong size\n";
            return 1;
        }

        // Start of inference
        // Forward pass timing start
        auto t0 = std::chrono::high_resolution_clock::now();

        // Input is single-channel [1,28,28]
        // Conv1: 1->6, k=5, pad=2 => [6,28,28]
        auto y1 = conv2d(
            x,                         // input [IC,H,W] flattened
            /*IC=*/1, /*H=*/28, /*W=*/28,
            W.conv1_w, W.conv1_b,
            /*OC=*/6, /*KH=*/5, /*KW=*/5, /*P=*/2
        );
        relu(y1);
        // Pool1: 2x2, stride 2 => [6,14,14]
        auto y1p = maxpool2x2(y1, /*C=*/6, /*H=*/28, /*W=*/28);

        // Conv2: 6->16, k=5, pad=0 => [16,10,10]
        auto y2 = conv2d(
            y1p,
            /*IC=*/6, /*H=*/14, /*W=*/14,
            W.conv2_w, W.conv2_b,
            /*OC=*/16, /*KH=*/5, /*KW=*/5, /*P=*/0
        );
        relu(y2);
        // Pool2: 2x2, stride 2 => [16,5,5]
        auto y2p = maxpool2x2(y2, /*C=*/16, /*H=*/10, /*W=*/10);

        // Flatten to 400
        std::vector<float> flat(y2p.begin(), y2p.end()); // size 16*5*5 = 400, std::vector range constructor

        // FC1: 400 -> 120
        auto h1 = linear(W.fc1_w, W.fc1_b, /*O=*/120, /*I=*/400, flat);
        relu(h1);

        // FC2: 120 -> 84
        auto h2 = linear(W.fc2_w, W.fc2_b, /*O=*/84, /*I=*/120, h1);
        relu(h2);

        // FC3: 84 -> 10 (logits)
        auto logits = linear(W.fc3_w, W.fc3_b, /*O=*/10, /*I=*/84, h2);

        // Forward pass timing end
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Argmax prediction
        int pred = 0;
        float best = logits[0];
        for (int i = 1; i < static_cast<int>(logits.size()); ++i) {
            if (logits[i] > best) {
                best = logits[i];
                pred = i;
            }
        }

        // Top-k
        auto probs = softmax(logits);
        int k = std::max(1, std::min(args.topk, static_cast<int>(logits.size())));
        std::vector<int> idx(logits.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](int a, int b) { return probs[a] > probs[b]; });

        // Output
        std::cout << "Prediction: " << pred << "\n";
        std::cout << "Top-" << k << " probabilities:\n";
        for (int i = 0; i < k; ++i) {
            int cls = idx[i];
            std::cout << "  class " << cls << ": " << probs[cls] << "\n";
        }
        std::cout << "Single inference time (forward pass): " << std::fixed << std::setprecision(3) << ms << " ms\n";

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}