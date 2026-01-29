//cv::equalizeHist function from OpeCV implemented in CUDA C++

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/find.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>

#include "utils.cuh"

struct is_positive
{
    __host__ __device__ bool operator() (int x) const { return x > 0; }
};

__global__ void equalizeHistKernel(const unsigned char* input, const unsigned char* mask, int* output, int w, int h) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int histogram[256];

    int threads = threadIdx.y * blockDim.x + threadIdx.x;
    int block  = blockDim.x * blockDim.y;

    for (int i = threads; i < 256; i += block) {
       histogram[i] = 0;
    }

    __syncthreads();

    if(col < w && row < h) {
       int index = row * w + col;
       if (mask[index] > 0) {
           unsigned char value = input[index];
           atomicAdd(&histogram[value], 1);
       }
    }

    __syncthreads();

    for (int i = threads; i < 256; i += block) {
       if (histogram[i] > 0) {
           atomicAdd(&output[i], histogram[i]);
       }
    }
}

__global__ void remapKernel(const unsigned char* input, const unsigned char* mask, const unsigned char* lut, unsigned char* output, int w, int h) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < w && row < h) {
        int idx = row * w + col;
        if (mask[idx] > 0) {
            output[idx] = lut[input[idx]];
        } else {
            output[idx] = input[idx];
        }
    }
}

torch::Tensor equalize_hist(torch::Tensor img, torch::Tensor mask) {
    TORCH_CHECK(img.device().is_cuda(), "img must be a CUDA tensor");
    TORCH_CHECK(mask.device().is_cuda(), "mask must be a CUDA tensor");

    const int height = img.size(0);
    const int width = img.size(1);
    const int bins = 256;

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(img.device());
    torch::Tensor hist_tensor = torch::zeros({bins}, options);
    int* d_hist = hist_tensor.data_ptr<int>();

    dim3 threads = getOptimalBlockDim(width, height);
    dim3 blocks(cdiv(width, threads.x), cdiv(height, threads.y));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    equalizeHistKernel<<<blocks, threads, 0, stream>>>(
        img.data_ptr<uint8_t>(),
        mask.data_ptr<uint8_t>(),
        d_hist,
        width,
        height
    );

    thrust::device_vector<int> CDF(bins);
    thrust::device_ptr<int> hist_ptr(d_hist);
    thrust::inclusive_scan(thrust::cuda::par.on(stream), hist_ptr, hist_ptr + bins, CDF.begin());

    auto iter_begin = CDF.begin();
    auto iter_end = CDF.end();
    auto iter2 = thrust::find_if(thrust::cuda::par.on(stream), iter_begin, iter_end, is_positive());

    int cdf_min = (iter2 != iter_end) ? (int)*iter2 : 0;

    thrust::device_ptr<uint8_t> mask_ptr(mask.data_ptr<uint8_t>());
    int active_pixels = thrust::count_if(thrust::cuda::par.on(stream),
                                         mask_ptr,
                                         mask_ptr + (width * height),
                                         [] __device__ (uint8_t m) { return m > 0; });

    auto lut_options = torch::TensorOptions().dtype(torch::kByte).device(img.device());
    torch::Tensor lut = torch::empty({bins}, lut_options);
    uint8_t* d_lut = lut.data_ptr<uint8_t>();

    thrust::transform(thrust::cuda::par.on(stream), CDF.begin(), CDF.end(), thrust::device_ptr<uint8_t>(d_lut),
       [=] __device__ (int cdf_val) {
           if (cdf_val == 0 || active_pixels <= cdf_min) return (uint8_t)0;
           float val = ((float)(cdf_val - cdf_min) / (float)(active_pixels - cdf_min)) * 255.0f;
           return (uint8_t)fminf(fmaxf(val, 0.0f), 255.0f);
       });

    torch::Tensor result = torch::empty_like(img);
    remapKernel<<<blocks, threads, 0, stream>>>(
        img.data_ptr<uint8_t>(),
        mask.data_ptr<uint8_t>(),
        d_lut,
        result.data_ptr<uint8_t>(),
        width,
        height
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}

//FastCV git:  https://github.com/JINO-ROHIT/fastcv
//OpenCV docs: https://docs.opencv.org/3.4/d4/d1b/tutorial_histogram_equalization.html