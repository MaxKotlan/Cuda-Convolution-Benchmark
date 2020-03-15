#include <iostream>
#include <fstream>
#include <random>
#include "Convolution.h"
#include "Benchmark.h"


std::ofstream ofs;
void Benchmark(){
    std::cout << "Starting Benchmark. Saving to KernelPerformance.csv" << std::endl;
    ofs.open ("KernelPerformance.csv", std::ofstream::out | std::ofstream::trunc);
    
    for (int i = 1; i < (1<<25); i*=2 ){
        std::vector<int> input(i);//(inputsize);
        std::generate(input.begin(), input.end(), []() { return rand()%100; });
        int filtersize = i/2 < 0x990 ? i/2 : 0x990;
        std::vector<int> filter(filtersize);
        std::generate(filter.begin(), filter.end(), []() { return rand()%100; });
        CsvPerformanceRow<int, 0x990>(ofs, true, input, filter);
    }
    
    ofs.close();
}