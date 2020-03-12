#include <iostream>
#include <cuda.h>
#include <time.h>
#include <chrono>
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

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;
typedef std::chrono::duration<float> fsec;

struct Startup{
    int threadsperblock = 1024;
} startup;

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

template <typename T>
const std::vector<ConvolutionCudaKernel<T>> cudaKernels{ 
    NaiveConvolution<T>, ConstantConvolution<T>, SharedConvolution<T> 
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
    algorithm<<< output.size() / startup.threadsperblock+1, startup.threadsperblock>>>(parameters);
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
    int rr = (vec.size()*2 > range) ? range : vec.size();
    int br = (vec.size()*2 > range) ? vec.size() - range/2 : vec.size();
    for (int i = 0; i < rr; i++)
        std::cout << vec[i] << ", ";
    std::cout << "... ";
    for (int i = br; i < vec.size(); i++)
        std::cout << vec[i] << ", ";
    std::cout << std::endl;
}

template<class T = int>
void printall(const std::vector<T>& vec) {
    for (auto e : vec)
        std::cout << e << ", ";
}

/*Tests the example in the lecture slides*/
void TestLectureExample(){
    /*Test Integer Convolution*/
    {
        std::vector<int> input{1,4,2,5};
        std::vector<int> filter{1,4,3};
        Result<int> r = CudaPerformConvolution(input, filter, NaiveConvolution);
        std::cout << "Testing: "; printall(input); std::cout << std::endl;
        std::cout << " Result: "; printall(r.output); std::cout << std::endl;
        assert(std::equal(r.output.begin(), r.output.end(), std::vector<int>{ 3, 16, 23, 27, 22, 5 }.begin() ));
    }

    /*Test Floating Point Convolution*/
    {
        std::vector<float> input{.5, 2., 1.,2.5};
        std::vector<float> filter{.5,.2,1.5};
        Result<float> r = CudaPerformConvolution(input, filter, NaiveConvolution);
        std::cout << "Testing: "; printall(input); std::cout << std::endl;
        std::cout << " Result: "; printall(r.output); std::cout << std::endl;
        assert(std::equal(r.output.begin(), r.output.end(), std::vector<float>{ 0.75, 3.1, 2.15, 4.95, 1, 1.25 }.begin() ));

    }
}

int main(int argc, char** argv){

    TestLectureExample();

    {
        std::vector<float> input{1,4,2,5};
        std::vector<float> filter{1,4,3,4};
        Result<float> r = CudaPerformConvolution(input, filter, NaiveConvolution);
        std::cout << "Testing: "; printall(input); std::cout << std::endl;
        std::cout << " Result: "; printall(r.output); std::cout << std::endl;
    }
    {
        int inputsize = 1024*1024*256;
        std::vector<int> input(inputsize);//(inputsize);
        std::generate(input.begin(), input.end(), []() { static int x = 0; x++;return x; });
        std::vector<int> filter(300);
        std::generate(filter.begin(), filter.end(), []() { static int x = -1; x++;return x; });

        std::cout << std::endl;
        Result<int> r1 = CudaPerformConvolution(input, filter, NaiveConvolution);
        std::cout << "Kernel Executed in: " << r1.executiontime << " milliseconds" << std::endl;
        printsome(r1.output, 10);
    }
    {
        int inputsize = 1024*1024*256;
        std::vector<float> input(inputsize);//(inputsize);
        std::generate(input.begin(), input.end(), []() { static float x = 0; x++;return x; });
        std::vector<float> filter(300);
        std::generate(filter.begin(), filter.end(), []() { static float x = -1.; x++;return x; });

        std::cout << std::endl;
        Result<float> r1 = CudaPerformConvolution(input, filter, NaiveConvolution);
        std::cout << "Kernel Executed in: " << r1.executiontime << " milliseconds" << std::endl;
        printsome(r1.output, 10);
        std::cout << std::endl;
    }
    {
        int inputsize = 1024;
        std::vector<float> input(inputsize);//(inputsize);
        std::generate(input.begin(), input.end(), []() { static float x = 0; x++;return x; });
        std::vector<float> filter(300);
        std::generate(filter.begin(), filter.end(), []() { static float x = -1.; x++;return x; });
        Result<float> r1 = CpuPerformConvolution(input, filter);
        std::cout << "Cpu Executed in: " << r1.executiontime << " milliseconds" << std::endl;
        printsome(r1.output, 10);
        std::cout << std::endl;
    }
}
