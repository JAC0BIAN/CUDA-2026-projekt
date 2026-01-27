//CUDA PROJECT sem.5 - cv::equalizeHist function from OpeCV implemented in CUDA C++
//Authors: Jakub Łukaszewski 197763, Artur Michna 197791
//Faculty od Electronics Telecommunication & Informatics, Gdańsk University of Technology 01.2026

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>

#include "utils.cuh"

// Kernel to compute histogram
__global__ void histogramKernel(unsigned char *in,
                                unsigned int *hist,
                                int w,
                                int h) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < w && row < h) {
        unsigned char pixel = in[row * w + col];
        atomicAdd(&hist[pixel], 1);
    }
}

// Kernel to apply the lookup table
__global__ void applyLUTKernel(unsigned char *in,
                               unsigned char *out,
                               unsigned char *lut,
                               int w,
                               int h) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < w && row < h) {
        int idx = row * w + col;
        out[idx] = lut[in[idx]];
    }
}

torch::Tensor equalize_hist(torch::Tensor img) {
    assert(img.device().type() == torch::kCUDA);
    assert(img.dtype() == torch::kByte);
    assert(img.dim() == 2);

    const auto height = img.size(0);
    const auto width = img.size(1);
    const int totalPixels = width * height;

    dim3 dimBlock = getOptimalBlockDim(width, height);
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y));

    // Allocate histogram on device using cudaMalloc
    unsigned int* d_hist;
    cudaMalloc(&d_hist, 256 * sizeof(unsigned int));
    cudaMemset(d_hist, 0, 256 * sizeof(unsigned int));

    // Compute histogram
    histogramKernel<<<dimGrid, dimBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
        img.data_ptr<unsigned char>(),
        d_hist,
        width,
        height
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Copy histogram to CPU
    std::vector<unsigned int> hist_cpu(256);
    cudaMemcpy(hist_cpu.data(), d_hist, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_hist);

    // Compute CDF and create LUT on CPU
    std::vector<unsigned char> lut(256);
    unsigned int cdf = 0;
    unsigned int cdf_min = 0;
    bool found_min = false;

    // First pass: find cdf_min
    for (int i = 0; i < 256; ++i) {
        cdf += hist_cpu[i];
        if (!found_min && cdf > 0) {
            cdf_min = cdf;
            found_min = true;
        }
    }

    // Second pass: compute LUT
    cdf = 0;
    for (int i = 0; i < 256; ++i) {
        cdf += hist_cpu[i];
        if (cdf == 0) {
            lut[i] = 0;
        } else {
            lut[i] = static_cast<unsigned char>(
                std::round((cdf - cdf_min) * 255.0 / (totalPixels - cdf_min))
            );
        }
    }

    // Transfer LUT to device
    auto lut_tensor = torch::from_blob(lut.data(), {256}, torch::kByte).to(img.device());

    // Apply LUT
    auto result = torch::empty({height, width},
                torch::TensorOptions().dtype(torch::kByte).device(img.device()));

    applyLUTKernel<<<dimGrid, dimBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
        img.data_ptr<unsigned char>(),
        result.data_ptr<unsigned char>(),
        lut_tensor.data_ptr<unsigned char>(),
        width,
        height
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}

//FastCV git:  https://github.com/JINO-ROHIT/fastcv
//OpenCV docs: https://docs.opencv.org/3.4/d4/d1b/tutorial_histogram_equalization.html