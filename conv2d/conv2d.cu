#include <iostream>
#include <stdio.h>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <assert.h>
#include <random>
#include <cstdlib>

#define MAX_KERNELS 128
#define MAX_THREAD_PER_BLOCK 1024
#define IMAGE_WIDTH 224
#define MAX_SHARED_FLT_ELEMENTS 8192

#define TILE_WIDTH 16
// warp size = 32, max sm shared memory is 64 KB or ~ 8192 

// __global__ void conv2d(const float* input, const float* output, const int Nx, const int Ny,
//                      const int Kx, const int Ky,
//                      const int Ni, const int Nn,
//                      const float* kernel){
//     __shared__ int s[(MAX_SHARED_FLT_ELEMENTS/IMAGE_WIDTH) - 1];
// }


#define checkCUDNN(expression) { \
    cudnnStatus_t status = (expression); \
    if (status != CUDNN_STATUS_SUCCESS){ \
        std::cerr << "cuDNN Error on line " << __LINE__ << ": " << cudnnGetErrorString(status) << std::endl; \
        std::exit(EXIT_FAILURE);                                \
    } \
}

#define checkCUDA(expression) { \
    cudaError_t status = (expression); \
    if (status != cudaSuccess){ \
        std::cerr << "CUDA Error on line " << __LINE__ << ": " << cudaGetErrorString(status) << std::endl; \
        std::exit(EXIT_FAILURE);                                \
    } \
}


bool checkAllocationProper(float* pointer_in){
    if (pointer_in == NULL){
        return false;
    }
    return true;
}

// ChatGPT Generated (Recommended in Prof.)
void runCudnnConv2D(
    float* d_input, float* d_kernel, float* d_output,
    int batch_size, int in_channels, int in_height, int in_width,
    int out_channels, int kernel_h, int kernel_w,
    int pad, int stride){
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    cudnnTensorDescriptor_t in_desc, out_desc;
    cudnnFilterDescriptor_t kernel_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    checkCUDNN(cudnnCreateTensorDescriptor(&in_desc));
    checkCUDNN(cudnnCreateTensorDescriptor(&out_desc));
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_desc));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

    checkCUDNN(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                          batch_size, in_channels, in_height, in_width));

    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                          out_channels, in_channels, kernel_h, kernel_w));

    checkCUDNN(cudnnSetConvolution2dDescriptor(conv_desc,
                                               pad, pad,
                                               stride, stride,
                                               1, 1,
                                               CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    int out_n, out_c, out_h, out_w;
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(conv_desc, in_desc, kernel_desc,
                                                     &out_n, &out_c, &out_h, &out_w));

    checkCUDNN(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                          out_n, out_c, out_h, out_w));


    cudnnConvolutionFwdAlgoPerf_t perf_results;
    int returnedAlgoCount = 0;
    
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        cudnn, in_desc, kernel_desc, conv_desc, out_desc,
        1, &returnedAlgoCount, &perf_results));
    
    cudnnConvolutionFwdAlgo_t algo;
    algo = perf_results.algo;

    size_t workspace_bytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                       in_desc,
                                                       kernel_desc,
                                                       conv_desc,
                                                       out_desc,
                                                       algo,
                                                       &workspace_bytes));

    void* d_workspace = nullptr;
    cudaMalloc(&d_workspace, workspace_bytes);

    const float alpha = 1.0f, beta = 0.0f;

    checkCUDNN(cudnnConvolutionForward(cudnn,
                                       &alpha,
                                       in_desc, d_input,
                                       kernel_desc, d_kernel,
                                       conv_desc,
                                       algo, d_workspace, workspace_bytes,
                                       &beta,
                                       out_desc, d_output));

    cudaFree(d_workspace);
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyFilterDescriptor(kernel_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);
}

__global__ void conv2d_naive(
    const float* __restrict__ input,    // [Ni × inH × inW]
    const float* __restrict__ kernel,   // [Nn × Ni × K_h × K_w]
    float* output,                      // [Nn × outH × outW]
    int Ni, int inH, int inW,
    int Nn,
    int K_h, int K_w,
    int pad, int stride)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_c = blockIdx.z;

    int outW = (inW - K_w + 2*pad)/stride + 1;
    int outH = (inH - K_h + 2*pad)/stride + 1;

    if (out_x >= outW || out_y >= outH || out_c >= Nn) return;

    float sum = 0.0f;
    for (int in_c = 0; in_c < Ni; ++in_c) {
        for (int ky = 0; ky < K_h; ++ky) {
            for (int kx = 0; kx < K_w; ++kx) {
                int in_x = out_x*stride + kx - pad;
                int in_y = out_y*stride + ky - pad;
                if (in_x>=0 && in_x<inW && in_y>=0 && in_y<inH) {
                    float v = input[(in_c*inH + in_y)*inW + in_x];
                    float w = kernel[((out_c*Ni + in_c)*K_h + ky)*K_w + kx];
                    sum += v*w;
                }
            }
        }
    }
    output[(out_c*outH + out_y)*outW + out_x] = sum;
}

int main(){
    /*
    Parameters
    */

    const int Nx=224, Ny=224, Kx=3, Ky=3, Ni=64, Nn=64, pad=0;     // Assume batch size = 1, and stride = 1
    const int stride=1, batch_size = 1;

    int numElements_in = Nx * Ny * Ni;
    int numElements_out = (Nx - Kx + 2*pad + stride) * (Ny - Ky + 2*pad + stride) * Nn / stride;
    int numElements_kernel = Kx * Ky * Ni * Nn;

    size_t in_size = numElements_in * sizeof(float);
    size_t out_size = numElements_out * sizeof(float);
    size_t kernel_size = numElements_kernel * sizeof(float);

    /*
    Host Memory Init.
    */

    float* h_in = (float *)malloc(in_size);
    float* h_kernel = (float *)malloc(kernel_size);
    float* h_out = (float *)malloc(out_size);

    if(!checkAllocationProper(h_in)) exit(EXIT_FAILURE);
    if(!checkAllocationProper(h_out)) exit(EXIT_FAILURE);
    if(!checkAllocationProper(h_kernel)) exit(EXIT_FAILURE);

    for (unsigned int i = 0; i < numElements_in; ++i){
        h_in[i] = rand() * 10;
    }
    for (unsigned int i = 0; i < numElements_kernel; ++i){
        h_kernel[i] = rand();
    }

    /*
    Device Memory Init. & Baseline Accuracy Generation (CUDA DNN)
    */

    float *d_kernel_BASELINE, *d_in_BASELINE, *d_out_BASELINE;

    cudaMalloc(&d_in_BASELINE, in_size);
    cudaMalloc(&d_out_BASELINE, out_size);
    cudaMalloc(&d_kernel_BASELINE, kernel_size);

    cudaMemcpy(d_in_BASELINE, h_in, in_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel_BASELINE, h_kernel, kernel_size, cudaMemcpyHostToDevice);

    runCudnnConv2D(
        d_in_BASELINE, d_kernel_BASELINE, d_out_BASELINE,
        1,              // batch_size
        64, 224, 224,   // Ni, H, W
        64, 3, 3,       // Nn, Kx, Ky
        0, 1            // padding, stride
    );

    


    // float *d_kernel_CUSTOM, *d_in_CUSTOM, *d_out_CUSTOM;

    // cudaMalloc(&d_in_CUSTOM, in_size);
    // cudaMalloc(&d_out_CUSTOM, out_size);
    // cudaMalloc(&d_kernel_CUSTOM, kernel_size);

    // cudaMemcpy(d_in_CUSTOM, h_in, in_size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_kernel_CUSTOM, h_kernel, kernel_size, cudaMemcpyHostToDevice);

    
    /* 
    Custom Kernel Init. and Function Call
    */
    //insert here


   


    checkCUDA(cudaFree(d_kernel_BASELINE)); checkCUDA(cudaFree(d_in_BASELINE)); checkCUDA(cudaFree(d_out_BASELINE));
    checkCUDA(cudaFree(d_kernel_CUSTOM)); checkCUDA(cudaFree(d_in_CUSTOM)); checkCUDA(cudaFree(d_out_CUSTOM));
    free(h_in); free(h_kernel); free(h_out);

    return 0;
}