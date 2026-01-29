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

struct is_positive //functor to find positive values
{
    __host__ __device__ bool operator() (int x) const { return x > 0; }
};

__global__ void equalizeHistKernel(const unsigned char* input, const unsigned char* mask, int* output, int w, int h) { //kerenel for his calculation
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int histogram[256]; //faster than global operations

    int threads = threadIdx.y * blockDim.x + threadIdx.x;
    int block  = blockDim.x * blockDim.y;

    for (int i = threads; i < 256; i += block) {
       histogram[i] = 0; //fill with 0's
    }

    __syncthreads(); // sync after thread operations

    if(col < w && row < h) { // prevent out of bounds operations
       int index = row * w + col;
       if (mask[index] > 0) {
           unsigned char value = input[index];
           atomicAdd(&histogram[value], 1); //avoids data corruption during multithread operations on the same bin
       }
    }

    __syncthreads();

    for (int i = threads; i < 256; i += block) {
		atomicAdd(&output[i], histogram[i]); //back to global
    }
}

__global__ void remapKernel(const unsigned char* input, const unsigned char* mask, const unsigned char* lut, unsigned char* output, int w, int h) { //kernel for applying mask
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < w && row < h) { // prevent out of bounds operations
        int idx = row * w + col; //2D -> 1D
        if (mask[idx] > 0) { // when there's a mask apply LUT
            output[idx] = lut[input[idx]];
        } else { // otherwise no change
            output[idx] = input[idx];
        }
    }
}

torch::Tensor equalize_hist(torch::Tensor img, torch::Tensor mask) {
    assert(img.device().type() == torch::kCUDA);
    assert(img.dtype() == torch::kByte);

    const int height = img.size(0);
    const int width = img.size(1);
    const int bins = 256;

	cudaStream_t stream;
    cudaStreamCreate(&stream);

	// Allocate histogram & fill with 0's
    int* hist;
	cudaMalloc(&hist, bins * sizeof(int));
	cudaMemsetAsync(hist, 0, bins * sizeof(int), stream);

    dim3 threads = getOptimalBlockDim(width, height);
    dim3 blocks(cdiv(width, threads.x), cdiv(height, threads.y)); // block dimensions

    equalizeHistKernel<<<blocks, threads, 0, stream>>>( // histogram kernel
        img.data_ptr<uint8_t>(),
        mask.data_ptr<uint8_t>(),
        hist,
        width,
        height
    );

	cudaStreamSynchronize(stream);

	// computing CDF from histogram
    thrust::device_vector<int> CDF(bins);
    thrust::device_ptr<int> hist_ptr(hist);
    thrust::inclusive_scan(thrust::cuda::par.on(stream), hist_ptr, hist_ptr + bins, CDF.begin()); //inclusive_scan ~= np.cumsum

	// find first nonzero value in CDF
	thrust::device_vector<int>::iterator iter1 = CDF.begin();
    thrust::device_vector<int>::iterator iter2 = thrust::find_if(CDF.begin(), CDF.end(), is_positive());

	int cdf_min = (iter2 != CDF.end()) ? *iter2 : 0;

    thrust::device_ptr<uint8_t> mask_ptr(mask.data_ptr<uint8_t>()); //uint8 is used for the equalizeHist algorythm
	//search for active pixels
    int active_pixels = thrust::count_if(thrust::cuda::par.on(stream),mask_ptr,mask_ptr + (width * height),[] __device__ (uint8_t m) { return m > 0; });

    unsigned char* lut;
	cudaMalloc(&lut, bins * sizeof(unsigned char));

    thrust::transform(thrust::cuda::par.on(stream), CDF.begin(), CDF.end(), thrust::device_ptr<uint8_t>(lut),
       [=] __device__ (int cdf_val) {
           if (cdf_val == 0 || active_pixels <= cdf_min) return (uint8_t)0; //if cdf is 0 then LUT = 0
           float val = ((float)(cdf_val - cdf_min) / (float)(active_pixels - cdf_min)) * 255.0f; // histogram equalization
           return (uint8_t)fminf(fmaxf(val, 0.0f), 255.0f);
       });

	cudaStreamSynchronize(stream);

	torch::Tensor result = torch::empty_like(img); //output tensor

    remapKernel<<<blocks, threads, 0, stream>>>(
        img.data_ptr<uint8_t>(),
        mask.data_ptr<uint8_t>(),
        lut,
        result.data_ptr<uint8_t>(),
        width,
        height
    );

	cudaStreamSynchronize(stream);

	cudaFree(hist);
	cudaFree(lut);
	cudaStreamDestroy(stream);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}

//FastCV git:  https://github.com/JINO-ROHIT/fastcv
//OpenCV docs: https://docs.opencv.org/3.4/d4/d1b/tutorial_histogram_equalization.html