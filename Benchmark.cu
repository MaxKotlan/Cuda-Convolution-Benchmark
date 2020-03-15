#include <iostream>
#include <fstream>
#include <random>
#include "Convolution.h"
#include "Benchmark.h"


std::ofstream ofs;
void Benchmark(){
    std::cout << "Starting Benchmark. Saving to KernelPerformance.csv" << std::endl;
    ofs.open ("KernelPerformance.csv", std::ofstream::out | std::ofstream::trunc);
    
    const int constantmemorysize = (0x9999 / (sizeof(int) +sizeof(float)));

    for (int i = 1; i < (1<<25); i*=2 ){

        std::vector<int> input(i);//(inputsize);
        std::generate(input.begin(), input.end(), []() { return rand()%100; });
        int filtersize = i/2 < constantmemorysize ? i/2 : constantmemorysize;
        std::vector<int> filter(filtersize);
        std::generate(filter.begin(), filter.end(), []() { return rand()%100; });

        std::vector<float> float_input(input.begin(), input.end());
        std::vector<float> float_filter(filter.begin(), filter.end());

        CsvPerformanceRow<int,   constantmemorysize>(ofs, true, false, input, filter);
        CsvPerformanceRow<float, constantmemorysize>(ofs, true, true, float_input, float_filter);
    }
    
    ofs.close();
}