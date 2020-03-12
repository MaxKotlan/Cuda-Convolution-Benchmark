#include <iostream>
#include <cuda.h>
#include <time.h>
#include <vector>
#include <algorithm>
#include <assert.h>

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
        executiontime = other.executiontime;
        output = std::move(other.output);
        return *this;
    }
};

struct KernelParameters {
    int* input;
    int  inputsize;
    int* filter;
    int  filtersize;
    int* output;
    int  outputsize;
    int  ghostvalue = 0;
};

__global__ void NaiveConvolution(KernelParameters parameters){
    int outputindex = blockDim.x * blockIdx.x + threadIdx.x;

    if (outputindex < parameters.outputsize){
        int result = parameters.ghostvalue;
        int inputstart = outputindex - (parameters.filtersize/2)-1;
        for (int filterindex = 0; filterindex < parameters.filtersize; filterindex++) {
            int inputindex = inputstart + filterindex;
            if (inputindex >= 0 && inputindex < parameters.outputsize)
                result += parameters.input[inputindex] * parameters.filter[filterindex];
        }
        parameters.output[outputindex] = result;
    }
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
    return inputsize+ filtersize-1;
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
        }
    }

    Result r = {0, input};
    return std::move(r);
}

Result CudaPerformConvolution(const std::vector<int>& input, const std::vector<int>& filter, ConvolutionCudaKernel algorithm){
    int* device_input, *device_filter, *device_output; Result result;
    std::vector<int> output(CalculateOutputSize(input.size(), filter.size()));

    gpuErrchk(cudaMalloc((void **)&device_input,   input.size()*sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&device_filter, filter.size()*sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&device_output, output.size()*sizeof(int)));

    gpuErrchk(cudaMemcpy(device_input,   input.data(),  input.size()*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device_filter, filter.data(), filter.size()*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device_output, output.data(), output.size()*sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);    

    KernelParameters parameters = { (int*)device_input, (int)input.size(), (int*)device_filter, (int)filter.size(), (int*)device_output, (int)output.size() };
    cudaEventRecord(start);
    algorithm<<< output.size() / startup.threadsperblock+1, startup.threadsperblock>>>(parameters);
    gpuErrchk(cudaEventRecord(stop));
    
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&result.executiontime, start, stop));

    gpuErrchk(cudaMemcpy(output.data(), device_output, output.size()*sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(device_input)); gpuErrchk(cudaFree(device_filter)); gpuErrchk(cudaFree(device_output));

    result.output = std::move(output);
    return std::move(result);
}

/*Prints a few elements from the front and a few from the back*/
void printsome(const std::vector<int>& vec, int range){
    int rr = (vec.size()*2 > range) ? range : vec.size();
    int br = (vec.size()*2 > range) ? vec.size() - range/2 : vec.size();
    for (int i = 0; i < rr; i++)
        std::cout << vec[i] << ", ";
    std::cout << "... ";
    for (int i = br; i < vec.size(); i++)
        std::cout << vec[i] << ", ";
    std::cout << std::endl;
}

void printall(const std::vector<int>& vec) {
    for (auto e : vec)
        std::cout << e << ", ";
}

/*Tests the example in the lecture slides*/
void TestLectureExample(){
    std::vector<int> input{1,4,2,5};
    std::vector<int> filter{1,4,3};

    Result r = CudaPerformConvolution(input, filter, NaiveConvolution);
    assert(std::equal(r.output.begin(), r.output.end(), std::vector<int>{ 3, 16, 23, 27, 22, 5 }.begin() ));

}

int main(int argc, char** argv){

    TestLectureExample();

    int inputsize = 1024*1024;
    std::vector<int> input(inputsize);//(inputsize);
    std::generate(input.begin(), input.end(), []() { static int x = 0; x++;return x; });
    std::vector<int> filter(101);
    std::generate(filter.begin(), filter.end(), []() { static int x = -1; x++;return x; });

    Result r = CpuPerformConvolution(input, filter);

    for (ConvolutionCudaKernel cudakern : cudaKernels){
        Result r1 = CudaPerformConvolution(input, filter, cudakern);
        std::cout << "Kernel Executed in: " << r1.executiontime << " milliseconds" << std::endl;
        printsome(r1.output, 10);
        std::cout << std::endl << std::endl;
    }
}
