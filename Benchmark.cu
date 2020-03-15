#include <iostream>
#include <fstream>
#include <random>
#include "Convolution.h"
#include "Benchmark.h"


void Benchmark(){
    std::cout << "Starting Benchmark. Saving to KernelPerformance.csv" << std::endl;
    std::ofstream ofs;
    ofs.open ("KernelPerformance.csv", std::ofstream::out | std::ofstream::trunc);
    
    /*Constant memory is fixed. To share between integer and float kernels, we must make sure the amount divides evenly into both*/
    /*On my device the maximum constant memory is 0x10000 bytes. This cannot be determined at runtime, because constant memory is calculated at compiletime*/
    const int constantmemorysize = (0x9999 / (sizeof(int) +sizeof(float)));

    /*Print Column Headers*/
    std::cout << "Input Size, Mask Size, ";
    ofs       << "Input Size, Mask Size, ";
    for (auto kernel : getKernels<int>()){
        std::cout << kernel.type << " " << kernel.label << ", ";
        ofs       << kernel.type << " " << kernel.label << ", ";
    }
    for (auto kernel : getKernels<float>()){
        std::cout << kernel.type << " " << kernel.label << ", ";
        ofs       << kernel.type << " " << kernel.label << ", ";
    }
    std::cout << std::endl;
    ofs       << std::endl;

    for (int i = 1; i < (1<<25); i*=2 ){

        /*Generates a random vector of size i, with a filter of size i/2. If i/2 > constantmemorysize 
        then use the maximum possible filtersize that will fit into const mem*/
        std::vector<int> input(i);//(inputsize);
        std::generate(input.begin(), input.end(), []() { return rand()%100; });
        int filtersize = i/2 < constantmemorysize ? i/2 : constantmemorysize;
        std::vector<int> filter(filtersize);
        std::generate(filter.begin(), filter.end(), []() { return rand()%100; });

        /*Ignore conversion warning with mvcc compiler*/
        #pragma warning(suppress: 4244)
        std::vector<float> float_input(input.begin(), input.end());
        std::vector<float> float_filter(filter.begin(), filter.end());

        std::cout << input.size() << ", " << filter.size() << ", ";
        ofs       << input.size() << ", " << filter.size() << ", ";


        /*Runs every cudakernel as an integer, with constant memory declared as constantmemorysize*/
        CsvPerformanceRow<int,   constantmemorysize>(ofs, input, filter);
        /*Runs every cudakernel as float, with constant memory declared as constantmemorysize*/
        CsvPerformanceRow<float, constantmemorysize>(ofs, float_input, float_filter);

        ofs << std::endl;
        std::cout << std::endl;
    }
    
    ofs.close();
}