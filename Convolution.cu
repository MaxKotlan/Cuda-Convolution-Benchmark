#include <iostream>
#include <cuda.h>
#include <time.h>
#include <vector>
#include <algorithm>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct Startup{
    int threadsperblock = 1024;
} startup;

struct Result {
    float executiontime;
    std::vector<int> output;

    /*Move operator to only shallow copy vector*/
    Result& operator=(const Result& other) {
        output = std::move(other.output);
        return *this;
    }
};

struct KernelParameters {
    int* input;
    int inputSize;
    int* kernel;
    int kernelSize;
    int* output;
    int outputSize;
};

__global__ void NaiveConvolution(KernelParameters parameters){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
}

__global__ void ConstantConvolution(KernelParameters parameters){
    
}

__global__ void SharedConvolution(KernelParameters parameters){
    
}

typedef void(*ConvolutionCudaKernel)(KernelParameters);
const std::vector<ConvolutionCudaKernel> cudaKernels{ 
    NaiveConvolution, ConstantConvolution, SharedConvolution 
};

bool isSymmetric(const std::vector<int>& vec){
    for(int i = 0; i < vec.size()/2; i++)
        if (vec[i] != vec[vec.size()-1-i])
            return false;
    return true;
}

int CalculateOutputSize(int inputsize, int filtersize){
    filtersize /= 2;
    return inputsize+ 2*filtersize;
}

Result CpuPerformConvolution(const std::vector<int>& input, const std::vector<int>& filter){
    std::vector<int> output(CalculateOutputSize(input.size(), filter.size()));

    bool isFilterSymmetric = isSymmetric(filter);
    for (int i = 0; i < input.size(); i++){
        if (isFilterSymmetric){
            for (int k = 0; k < filter.size()/2+1; k++){
                output[k+i] = filter[k] * input[k+i];
                output[(filter.size() - k - 1)+i] = filter[k] * input[(filter.size() - k - 1)+i];
            }
        } else {
            for (int k = 0; k < filter.size(); k++){

            }
        }
    }

    Result r = {0, input};
    return std::move(r);
}

Result CudaPerformConvolution(const std::vector<int>& input, const std::vector<int>& kernel, ConvolutionCudaKernel algorithm){
    int* device_input, *device_kernel, *device_output; Result result;
    std::vector<int> output(CalculateOutputSize(input.size(), kernel.size()));

    gpuErrchk(cudaMalloc((void **)&device_input,   input.size()*sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&device_kernel, kernel.size()*sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&device_output, output.size()*sizeof(int)));

    gpuErrchk(cudaMemcpy(device_input,   input.data(),  input.size()*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device_kernel, kernel.data(), kernel.size()*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device_output, output.data(), output.size()*sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);    

    KernelParameters parameters = { (int*)device_input, (int)input.size(), (int*)device_kernel, (int)kernel.size(), (int*)device_output, (int)output.size() };
    algorithm<<< output.size() / startup.threadsperblock+1, startup.threadsperblock>>>(parameters);

    cudaEventRecord(stop);
    gpuErrchk(cudaEventSynchronize(stop));
    cudaEventElapsedTime(&result.executiontime, start, stop);

    gpuErrchk(cudaMemcpy(output.data(), device_output, output.size()*sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(device_input)); gpuErrchk(cudaFree(device_kernel)); gpuErrchk(cudaFree(device_output));

    result.output = std::move(output);
    return std::move(result);
}

int main(int argc, char** argv){
    int inputsize = 512;
    std::vector<int> input(inputsize);
    std::generate(input.begin(), input.end(), []() { static int x = -1; x++;return x; });
    std::vector<int> filter{1,2,3,2,1};

    Result r = CpuPerformConvolution(input, filter);

    for (ConvolutionCudaKernel cudakern : cudaKernels){
        Result r1 = CudaPerformConvolution(input, filter, cudakern);
        std::cout << "Kernel Executed in: " << r1.executiontime << " milliseconds" << std::endl;
        for (auto element : r1.output)
            std::cout << element << ", ";
        std::cout << std::endl << std::endl;
    }
}
