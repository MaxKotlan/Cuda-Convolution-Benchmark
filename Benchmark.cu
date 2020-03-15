#include <iostream>
#include <fstream>
#include <random>
#include "Convolution.h"
#include "Benchmark.h"


std::fstream fs;

void Benchmark(){
    std::cout << "Starting Benchmark. Saving to KernelPerformance.csv" << std::endl;
    fs.open ("KernelPerformance.csv", std::fstream::in | std::fstream::out | std::fstream::app);
    
    for (int i = 1; i < (1<<25); i*=2 ){
        std::vector<int> input(i);//(inputsize);
        std::generate(input.begin(), input.end(), []() { return rand()%100; });
        std::vector<int> filter(i/2);
        std::generate(filter.begin(), filter.end(), []() { return rand()%100; });
        CsvPerformanceRow<int, 0x990>(fs, true, input, filter);
    }
    
    fs.close();
}