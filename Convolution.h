#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <iostream>
#include <cuda.h>
#include <time.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <assert.h>
#include "Startup.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;
typedef std::chrono::duration<float> fsec;


template<class T = int>
struct Result {
    float executiontime;
    std::vector<T> output;

    /*Move operator to only shallow copy vector*/
    Result& operator=(const Result& other) {
        executiontime = other.executiontime;
        output = std::move(other.output);
        return *this;
    }
};

template<class T = int>
struct KernelParameters {
    T* input;
    int  inputsize;
    T* filter;
    int  filtersize;
    T* output;
    int  outputsize;
    T  ghostvalue = (T)0;
};

template<class T = int>
__global__ void NaiveConvolution(KernelParameters<T> parameters){
    int outputindex = blockDim.x * blockIdx.x + threadIdx.x;

    if (outputindex < parameters.outputsize){
        T result = parameters.ghostvalue;
        int inputstart = outputindex - (parameters.filtersize/2)-1;
        for (int filterindex = 0; filterindex < parameters.filtersize; filterindex++) {
            int inputindex = inputstart + filterindex;
            if (inputindex >= 0 && inputindex < parameters.outputsize-2)
                result += parameters.input[inputindex] * parameters.filter[filterindex];
        }
        parameters.output[outputindex] = result;
    }
}

template<class T = int>
__global__ void ConstantConvolution(KernelParameters<T> parameters){
    
}

template<class T = int>
__global__ void SharedConvolution(KernelParameters<T> parameters){
    
}

template <class T>
using ConvolutionCudaKernel = void(*)(KernelParameters<T>);

template <class T>
const std::vector<ConvolutionCudaKernel<T>> CudaKernels{ 
    NaiveConvolution<T>, ConstantConvolution<T>, SharedConvolution<T> 
};

template <class T>
bool isSymmetric(const std::vector<T>& vec){
    for(int i = 0; i < vec.size()/2; i++)
        if (vec[i] != vec[vec.size()-1-i])
            return false;
    return true;
}

template<class T = int>
T CalculateOutputSize(T inputsize, T filtersize){
    return inputsize+ filtersize-1;
}

template<class T = int>
Result<T> CpuPerformConvolution(const std::vector<T>& input, const std::vector<T>& filter){
    std::vector<T> output(CalculateOutputSize(input.size(), filter.size()));

    auto t0 = Time::now();
    for (int outputindex = 0; outputindex < output.size(); outputindex++){
        T result = 0;
        int inputstart = outputindex - (filter.size()/2)-1;
        for (int filterindex = 0; filterindex < filter.size(); filterindex++) {
            int inputindex = inputstart + filterindex;
            if (inputindex >= 0 && inputindex < output.size()-2)
                result += input[inputindex] * filter[filterindex];
        }
        output[outputindex] = result;
    }
    auto t1 = Time::now();
    fsec fs = t1 - t0;
    ms d = std::chrono::duration_cast<ms>(fs);

    Result<T> r = {(float)d.count(), std::move(output)};
    return std::move(r);
}

template<class T = int>
Result<T> CudaPerformConvolution(const std::vector<T>& input, const std::vector<T>& filter, ConvolutionCudaKernel<T> algorithm){
    T* device_input, *device_filter, *device_output; Result<T> result;
    std::vector<T> output(CalculateOutputSize(input.size(), filter.size()));

    gpuErrchk(cudaMalloc((void **)&device_input,   input.size()*sizeof(T)));
    gpuErrchk(cudaMalloc((void **)&device_filter, filter.size()*sizeof(T)));
    gpuErrchk(cudaMalloc((void **)&device_output, output.size()*sizeof(T)));

    gpuErrchk(cudaMemcpy(device_input,   input.data(),  input.size()*sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device_filter, filter.data(), filter.size()*sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device_output, output.data(), output.size()*sizeof(T), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);    

    KernelParameters<T> parameters = { (T*)device_input, (int)input.size(), (T*)device_filter, (int)filter.size(), (T*)device_output, (int)output.size() };
    cudaEventRecord(start);
    algorithm<<< output.size() / 1024+1, 1024>>>(parameters);
    gpuErrchk(cudaEventRecord(stop));
    
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&result.executiontime, start, stop));

    gpuErrchk(cudaMemcpy(output.data(), device_output, output.size()*sizeof(T), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(device_input)); gpuErrchk(cudaFree(device_filter)); gpuErrchk(cudaFree(device_output));

    result.output = std::move(output);
    return std::move(result);
}

/*Prints a few elements from the front and a few from the back*/
template<class T = int>
void printsome(const std::vector<T>& vec, int range){
    bool isSmallVector = vec.size() > range;
    int rr = isSmallVector ? range : vec.size();
    int br = isSmallVector ? vec.size() - range/2 : vec.size();
    for (int i = 0; i < rr; i++)
        std::cout << vec[i] << ", ";
    if (!isSmallVector) std::cout << "... ";
    for (int i = br; i < vec.size(); i++)
        std::cout << vec[i] << ", ";
}

template<class T = int>
void printall(const std::vector<T>& vec) {
    for (auto e : vec)
        std::cout << e << ", ";
} 

/*Test Visually*/
template<class T = int>
Result<T> Test(const std::vector<T>& input, const std::vector<T>& filter){
    std::cout << "Testing Input: "; printsome(input,10);  std::cout << std::endl;
    std::cout << "With Filter:   "; printsome(filter,10); std::cout << std::endl;
    Result<T> r = CudaPerformConvolution(input, filter, NaiveConvolution);
    std::cout << "Result Vector: "; printsome(r.output,10); std::cout << std::endl;
    return std::move(r);
}

/*Test and Assert*/
template<class T = int>
void Test(const std::vector<T>& input, const std::vector<T>& filter, const std::vector<T>& expected){
    Result<T> r = Test(input, filter);
    assert(std::equal(r.output.begin(), r.output.end(), expected.begin() ));
}

#endif