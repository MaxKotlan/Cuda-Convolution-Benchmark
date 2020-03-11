#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda.h>

int main(int argc, char** argv){
    
    int size = 1024;
    std::vector<int> input(size);
    std::generate(input.begin(), input.end(), []() { return rand() % 100; });
}
