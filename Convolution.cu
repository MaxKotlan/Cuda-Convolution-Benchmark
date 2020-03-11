#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <cuda.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

typedef struct Startup{
    unsigned int threadsperblock = 1024;
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

__global__ void NaiveConvolution(int* data, int dataSize, int* kernel, int kernelSize, int* output, int outputSize){

}

__global__ void ConstantConvolution(int* data, int dataSize, int* kernel, int kernelSize, int* output, int outputSize){
    
}

__global__ void SharedConvolution(int* data, int dataSize, int* kernel, int kernelSize, int* output, int outputSize){
    
}

typedef std::function<void(int*,int,int*,int,int*,int)> ConvolutionCudaKernel;
std::vector<ConvolutionCudaKernel> cudaKernels{ 
    NaiveConvolution, ConstantConvolution, SharedConvolution 
};

bool isSymmetric(const std::vector<int>& vec){
    for(int i = 0; i < vec.size()/2; i++)
        if (vec[i] != vec[vec.size()-1-i])
            return false;
    return true;
}

int CalculateOutputSize(int inputsize, int kernelsize){
    kernelsize /= 2;
    return inputsize+ 2*kernelsize;
}

Result CpuPerformConvolution(const std::vector<int>& input, const std::vector<int>& kernel){
    std::vector<int> output(CalculateOutputSize(input.size(), kernel.size()));

    bool isKernelSymmetric = isSymmetric(kernel);
    for (int i = 0; i < input.size(); i++){
        if (isKernelSymmetric){
            for (int k = 0; k < kernel.size()/2+1; k++){
                output[k+i] = kernel[k] * input[k+i];
                output[(kernel.size() - k - 1)+i] = kernel[k] * input[(kernel.size() - k - 1)+i];
            }
        } else {
            for (int k = 0; k < kernel.size(); k++){

            }
        }
    }

    Result r = {0, input};
    return std::move(r);
}

Result CudaPerformConvolution(const std::vector<int>& input, const std::vector<int>& kernel, ConvolutionCudaKernel cudakernel){
    int* device_input, *device_kernel, *device_output;
    std::vector<int> output(CalculateOutputSize(input.size(), kernel.size()));

    gpuErrchk(cudaMalloc((void **)&device_input,   input.size()));
    gpuErrchk(cudaMalloc((void **)&device_kernel, kernel.size()));
    gpuErrchk(cudaMalloc((void **)&device_output, output.size()));

    gpuErrchk(cudaMemcpy(device_input,   input.data(),  input.size(), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device_kernel, kernel.data(), kernel.size(), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device_output, output.data(), output.size(), cudaMemcpyHostToDevice));

    unsigned int threadsneeded = 0;
    //cudakernel<<<output.size() / threadsperblock+1, threadsperblock>>>(device_input, device_kernel, device_output)

    Result r = {0, input};
    return std::move(r);
}

int main(int argc, char** argv){
    
    int inputsize = 1024; int kernelsize = 3;
    std::vector<int> input(inputsize);
    std::generate(input.begin(), input.end(), []() { return rand() % 100; });
    std::vector<int> kernel{1,2,3,2,1};
    Result r = CpuPerformConvolution(input, kernel);
}
