#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <cstring>


static void checkErr(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL error " << err << " at " << msg << std::endl;
        std::exit(1);
    }
}

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

    // OpenCL setup: platform, device, context, queue
    cl_int err;
    // Get all platforms. Assuming 2 here.
    cl_platform_id platform[2]; // assuming a total of 2 platforms.
    err = clGetPlatformIDs(2, platform, NULL);
    checkErr(err, "clGetPlatformIDs failed");

    // Get CPU/GPU/FPGA device
    // Change CL_DEVICE_TYPE_SEL to CL_DEVICE_TYPE_CPU if using PoCL, CL_DEVICE_TYPE_GPU if using GPU.
    // For PC, GPU is generally the first platform (PLATFORM_INDEX is 0), CPU (PoCL) second. 
    // For Kria, CPU (PoCL, if installed) is generally the first platform (PLATFORM_INDEX is 0), FPGA second.
    // **Important**: Check the order via `clinfo` and make selections appropropriately.
    #ifdef USE_XCLBIN
        #define CL_DEVICE_TYPE_SEL CL_DEVICE_TYPE_ACCELERATOR
        #define PLATFORM_INDEX 1
    #else
        #define CL_DEVICE_TYPE_SEL CL_DEVICE_TYPE_CPU
        #define PLATFORM_INDEX 0 
    #endif
    
    cl_device_id device;
    err = clGetDeviceIDs(platform[PLATFORM_INDEX], CL_DEVICE_TYPE_SEL, 1, &device, NULL); 
    checkErr(err, "clGetDeviceIDs failed");

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    checkErr(err, "clCreateContext");
    // cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    checkErr(err, "clCreateCommandQueue");

    // Build program
    auto kernel_src = readFileBytes(kernel_path);
    const char* src = kernel_src.data();
    size_t src_len = kernel_src.size();
    cl_program program = clCreateProgramWithSource(context, 1, &src, &src_len, &err);
    checkErr(err, "clCreateProgramWithSource");
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Build log:\n" << log.data() << std::endl;
        checkErr(err, "clBuildProgram");
    }

    // Create kernels
    cl_kernel k_conv = clCreateKernel(program, "conv2d_relu", &err); checkErr(err, "clCreateKernel conv2d_relu");
    cl_kernel k_pool = clCreateKernel(program, "maxpool2x2", &err);  checkErr(err, "clCreateKernel maxpool2x2");
    cl_kernel k_fc1  = clCreateKernel(program, "fc_relu", &err);     checkErr(err, "clCreateKernel fc_relu");
    cl_kernel k_fc2  = clCreateKernel(program, "fc_relu", &err);     checkErr(err, "clCreateKernel fc_relu");
    cl_kernel k_fc3  = clCreateKernel(program, "fc_norelu", &err);   checkErr(err, "clCreateKernel fc_norelu");

    // Allocate device buffers for weights/biases
    auto mkbuf = [&](size_t nBytes, const void* host)->cl_mem {
        cl_mem b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nBytes, const_cast<void*>(host), &err);
        checkErr(err, "clCreateBuffer weights");
        return b;
    };
    cl_mem d_conv1_w = mkbuf(conv1_w.size()*sizeof(float), conv1_w.data());
    cl_mem d_conv1_b = mkbuf(conv1_b.size()*sizeof(float), conv1_b.data());
    cl_mem d_conv2_w = mkbuf(conv2_w.size()*sizeof(float), conv2_w.data());
    cl_mem d_conv2_b = mkbuf(conv2_b.size()*sizeof(float), conv2_b.data());
    cl_mem d_fc1_w   = mkbuf(fc1_w.size()*sizeof(float), fc1_w.data());
    cl_mem d_fc1_b   = mkbuf(fc1_b.size()*sizeof(float), fc1_b.data());
    cl_mem d_fc2_w   = mkbuf(fc2_w.size()*sizeof(float), fc2_w.data());
    cl_mem d_fc2_b   = mkbuf(fc2_b.size()*sizeof(float), fc2_b.data());
    cl_mem d_fc3_w   = mkbuf(fc3_w.size()*sizeof(float), fc3_w.data());
    cl_mem d_fc3_b   = mkbuf(fc3_b.size()*sizeof(float), fc3_b.data());

    // Intermediate feature maps
    const int C1 = 6, C2 = 16;
    std::vector<float> conv1_out(C1*28*28);
    std::vector<float> pool1_out(C1*14*14);
    std::vector<float> conv2_out(C2*10*10);
    std::vector<float> pool2_out(C2*5*5);
    std::vector<float> fc1_out(120);
    std::vector<float> fc2_out(84);
    std::vector<float> fc3_out(10);

    cl_mem d_fc1_in = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 400, nullptr, &err); checkErr(err, "fc1_in");

    cl_mem d_input   = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*28*28, nullptr, &err); checkErr(err, "clCreateBuffer input");
    cl_mem d_conv1_o = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*conv1_out.size(), nullptr, &err); checkErr(err, "conv1_o");
    cl_mem d_pool1_o = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*pool1_out.size(), nullptr, &err); checkErr(err, "pool1_o");
    cl_mem d_conv2_o = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*conv2_out.size(), nullptr, &err); checkErr(err, "conv2_o");
    cl_mem d_pool2_o = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*pool2_out.size(), nullptr, &err); checkErr(err, "pool2_o");
    cl_mem d_fc1_o   = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*fc1_out.size(), nullptr, &err);    checkErr(err, "fc1_o");
    cl_mem d_fc2_o   = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*fc2_out.size(), nullptr, &err);    checkErr(err, "fc2_o");
    cl_mem d_fc3_o   = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*fc3_out.size(), nullptr, &err);    checkErr(err, "fc3_o");

    auto set_conv_args = [&](cl_mem d_in, cl_mem d_w, cl_mem d_b, int in_c, int out_c, int in_h, int in_w, int k, int stride, int pad, int out_h, int out_w, cl_mem d_out){
        int a = 0;
        checkErr(clSetKernelArg(k_conv, a++, sizeof(cl_mem), &d_in), "set arg");
        checkErr(clSetKernelArg(k_conv, a++, sizeof(cl_mem), &d_w), "set arg");
        checkErr(clSetKernelArg(k_conv, a++, sizeof(cl_mem), &d_b), "set arg");
        checkErr(clSetKernelArg(k_conv, a++, sizeof(int), &in_c), "set arg");
        checkErr(clSetKernelArg(k_conv, a++, sizeof(int), &out_c), "set arg");
        checkErr(clSetKernelArg(k_conv, a++, sizeof(int), &in_h), "set arg");
        checkErr(clSetKernelArg(k_conv, a++, sizeof(int), &in_w), "set arg");
        checkErr(clSetKernelArg(k_conv, a++, sizeof(int), &k), "set arg");
        checkErr(clSetKernelArg(k_conv, a++, sizeof(int), &stride), "set arg");
        checkErr(clSetKernelArg(k_conv, a++, sizeof(int), &pad), "set arg");
        checkErr(clSetKernelArg(k_conv, a++, sizeof(int), &out_h), "set arg");
        checkErr(clSetKernelArg(k_conv, a++, sizeof(int), &out_w), "set arg");
        checkErr(clSetKernelArg(k_conv, a++, sizeof(cl_mem), &d_out), "set arg");
    };
    auto set_pool_args = [&](cl_mem d_in, int channels, int in_h, int in_w, cl_mem d_out){
        int a = 0;
        checkErr(clSetKernelArg(k_pool, a++, sizeof(cl_mem), &d_in), "set arg");
        checkErr(clSetKernelArg(k_pool, a++, sizeof(int), &channels), "set arg");
        checkErr(clSetKernelArg(k_pool, a++, sizeof(int), &in_h), "set arg");
        checkErr(clSetKernelArg(k_pool, a++, sizeof(int), &in_w), "set arg");
        checkErr(clSetKernelArg(k_pool, a++, sizeof(cl_mem), &d_out), "set arg");
    };
    auto set_fc_args = [&](cl_kernel k, cl_mem d_in, cl_mem d_w, cl_mem d_b, int in_size, int out_size, cl_mem d_out){
        int a = 0;
        checkErr(clSetKernelArg(k, a++, sizeof(cl_mem), &d_in), "set arg");
        checkErr(clSetKernelArg(k, a++, sizeof(cl_mem), &d_w), "set arg");
        checkErr(clSetKernelArg(k, a++, sizeof(cl_mem), &d_b), "set arg");
        checkErr(clSetKernelArg(k, a++, sizeof(int), &in_size), "set arg");
        checkErr(clSetKernelArg(k, a++, sizeof(int), &out_size), "set arg");
        checkErr(clSetKernelArg(k, a++, sizeof(cl_mem), &d_out), "set arg");
    };

    size_t correct = 0;
    double total_ms = 0.0;

    for (int idx = 0; idx < num_images; idx++) {
        const float* img = images.data() + idx * 28 * 28;

        // Upload input image
        checkErr(clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, sizeof(float)*28*28, img, 0, nullptr, nullptr), "write input");

        auto t0 = std::chrono::high_resolution_clock::now();

        // conv1: in 1x28x28 -> out 6x28x28 (k5, s1, pad2)
        set_conv_args(d_input, d_conv1_w, d_conv1_b, 1, 6, 28, 28, 5, 1, 2, 28, 28, d_conv1_o);
        size_t g_conv1 = (size_t)(6 * 28 * 28);
        checkErr(clEnqueueNDRangeKernel(queue, k_conv, 1, nullptr, &g_conv1, nullptr, 0, nullptr, nullptr), "enqueue conv1");

        // pool1: 6x28x28 -> 6x14x14
        set_pool_args(d_conv1_o, 6, 28, 28, d_pool1_o);
        size_t g_pool1 = (size_t)(6 * 14 * 14);
        checkErr(clEnqueueNDRangeKernel(queue, k_pool, 1, nullptr, &g_pool1, nullptr, 0, nullptr, nullptr), "enqueue pool1");

        // conv2: in 6x14x14 -> out 16x10x10 (k5,s1,pad0)
        set_conv_args(d_pool1_o, d_conv2_w, d_conv2_b, 6, 16, 14, 14, 5, 1, 0, 10, 10, d_conv2_o);
        size_t g_conv2 = (size_t)(16 * 10 * 10);
        checkErr(clEnqueueNDRangeKernel(queue, k_conv, 1, nullptr, &g_conv2, nullptr, 0, nullptr, nullptr), "enqueue conv2");

        // pool2: 16x10x10 -> 16x5x5
        set_pool_args(d_conv2_o, 16, 10, 10, d_pool2_o);
        size_t g_pool2 = (size_t)(16 * 5 * 5);
        checkErr(clEnqueueNDRangeKernel(queue, k_pool, 1, nullptr, &g_pool2, nullptr, 0, nullptr, nullptr), "enqueue pool2");

        // Read pool2 to host to flatten for FCs
        checkErr(clEnqueueReadBuffer(queue, d_pool2_o, CL_TRUE, 0, sizeof(float)*pool2_out.size(), pool2_out.data(), 0, nullptr, nullptr), "read pool2");

        // Upload flattened to d_fc1 input buffer
        checkErr(clEnqueueWriteBuffer(queue, d_fc1_in, CL_TRUE, 0, sizeof(float)*pool2_out.size(), pool2_out.data(), 0, nullptr, nullptr), "write fc1 input");
        
        // fc1: 400 -> 120 (ReLU)
        // set_fc_args(k_fc1, d_fc1_o, d_fc1_w, d_fc1_b, 400, 120, d_fc1_o);
        set_fc_args(k_fc1, d_fc1_in, d_fc1_w, d_fc1_b, 400, 120, d_fc1_o);
        size_t g_fc1 = 120;
        checkErr(clEnqueueNDRangeKernel(queue, k_fc1, 1, nullptr, &g_fc1, nullptr, 0, nullptr, nullptr), "enqueue fc1");

        // fc2: 120 -> 84 (ReLU)
        set_fc_args(k_fc2, d_fc1_o, d_fc2_w, d_fc2_b, 120, 84, d_fc2_o);
        size_t g_fc2 = 84;
        checkErr(clEnqueueNDRangeKernel(queue, k_fc2, 1, nullptr, &g_fc2, nullptr, 0, nullptr, nullptr), "enqueue fc2");

        // fc3: 84 -> 10 (no ReLU)
        set_fc_args(k_fc3, d_fc2_o, d_fc3_w, d_fc3_b, 84, 10, d_fc3_o);
        size_t g_fc3 = 10;
        checkErr(clEnqueueNDRangeKernel(queue, k_fc3, 1, nullptr, &g_fc3, nullptr, 0, nullptr, nullptr), "enqueue fc3");

        checkErr(clFinish(queue), "clFinish");

        auto t1 = std::chrono::high_resolution_clock::now();

        // Read logits
        checkErr(clEnqueueReadBuffer(queue, d_fc3_o, CL_TRUE, 0, sizeof(float)*fc3_out.size(), fc3_out.data(), 0, nullptr, nullptr), "read fc3");

        int pred = 0;
        float maxv = fc3_out[0];
        for (int i = 1; i < 10; i++) {
            if (fc3_out[i] > maxv) { maxv = fc3_out[i]; pred = i; }
        }
        if (pred == (int)labels[idx]) correct++;

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        if (idx < 10) {
            std::cout << "Image " << idx << ": Prediction: " << pred << ", True Label: " << (int)labels[idx] << std::endl;
        }
    }

    double avg_ms = total_ms / num_images;
    double fps = 1000.0 / avg_ms;
    std::cout << "\n=== Results (PoCL CPU) ===\n";
    std::cout << "Accuracy: " << (100.0 * correct / num_images) << "% (" << correct << "/" << num_images << ")\n";
    std::cout << "Average inference time: " << avg_ms << " ms\n";
    std::cout << "Throughput: " << fps << " FPS\n";

    // Cleanup
    clReleaseMemObject(d_conv1_w); clReleaseMemObject(d_conv1_b);
    clReleaseMemObject(d_conv2_w); clReleaseMemObject(d_conv2_b);
    clReleaseMemObject(d_fc1_w);   clReleaseMemObject(d_fc1_b);
    clReleaseMemObject(d_fc2_w);   clReleaseMemObject(d_fc2_b);
    clReleaseMemObject(d_fc3_w);   clReleaseMemObject(d_fc3_b);
    clReleaseMemObject(d_input);   clReleaseMemObject(d_conv1_o);
    clReleaseMemObject(d_pool1_o); clReleaseMemObject(d_conv2_o);
    clReleaseMemObject(d_pool2_o); clReleaseMemObject(d_fc1_o);
    clReleaseMemObject(d_fc2_o);   clReleaseMemObject(d_fc3_o);
    clReleaseMemObject(d_fc1_in);
    clReleaseKernel(k_conv); clReleaseKernel(k_pool);
    clReleaseKernel(k_fc1);  clReleaseKernel(k_fc2); clReleaseKernel(k_fc3);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}