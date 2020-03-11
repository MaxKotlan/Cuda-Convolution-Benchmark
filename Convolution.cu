#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <cuda.h>

struct Startup{

};

struct Result {
    float executiontime;
    std::vector<int> output;

    /*Move operator to only shallow copy vector*/
    Result& operator=(const Result& other) {
        output = std::move(other.output);
        return *this;
    }
};

//__global__ NaiveConvolution(int* data, int datasize, int* kernel, int* kernelsize){
//
//
//}

bool Symmetric(const std::vector<int>& vec){
    for(int i = 0; i < vec.size()/2; i++)
        if (vec[i] != vec[vec.size()-1-i])
            return false;
    return true;
}

int CalculateOutputSize(inputsize, kernelsize){
    kernelsize /= 2;
    return inputsize+ 2*kernelsize;
}

Result CpuPerformConvolution(const std::vector<int>& input, const std::vector<int>& kernel){
    std::vector<int> output(CalculateOutputSize(input.size(), kernel.size()));

    bool isKernelSymmetric = Symmetric(kernel);
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

Result CudaPerformConvolution(const std::vector<int>& input, const std::vector<int>& kernel){
    Result r = {0, input};
    return std::move(r);
}

int main(int argc, char** argv){
    
    int inputsize = 1024; int kernelsize = 3;
    std::vector<int> input(inputsize);
    std::generate(input.begin(), input.end(), []() { return rand() % 100; });
    std::vector<int> kernel{0,1,0};
    Result r = PerformConvolution(input, kernel);
}
