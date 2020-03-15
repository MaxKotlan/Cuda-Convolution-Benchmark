#include "Convolution.h"
#include "Test.h"

const int headerlength = 52;

void printHeader(std::string label){
    std::cout << "-" << label; for (int i = 0; i < headerlength - label.size()-1; i++) std::cout << "-"; std::cout << std::endl;
}
void printFooter(){ std::cout << std::endl; }

void RunTests(){
    /*Testing Convolution Example from Slide as an Integr and a float*/
    /*Input, Filter, Expected Value*/
    
    printHeader("Trivial");
        TestAllKernels(std::vector<int>  {0}, std::vector<int>  {0}, std::vector<int>  {0});
        TestAllKernels(std::vector<float>{0}, std::vector<float>{0}, std::vector<float>{0});

        TestAllKernels(std::vector<float>{0,0}, std::vector<float>{1,1,1}, std::vector<float>{0,0,0,0});
        TestAllKernels(std::vector<float>{1,1}, std::vector<float>{0,0,0}, std::vector<float>{0,0,0,0});

        TestAllKernels(std::vector<float>{1,1}, std::vector<float>{1,1}, std::vector<float>{1,2,1});
        TestAllKernels(std::vector<float>{1,1}, std::vector<float>{1,2}, std::vector<float>{2,3,1});
        TestAllKernels(std::vector<float>{1,1,1,1,1,1,1}, std::vector<float>{1}, std::vector<float>{1,1,1,1,1,1,1});

    printFooter();
    printHeader("Lecture Slide");
    {
        TestAllKernels(std::vector<int>  {1,4,2,5}, std::vector<int>  {1,4,3},   std::vector<int>  {3,16,23,27,22,5});
        TestAllKernels(std::vector<float>{1,4,2,5}, std::vector<float>{1,4,3},   std::vector<float>{3,16,23,27,22,5});
        std::cout << "\tTesting cpu version: ";
        Result<int> cpures = CpuPerformConvolution(std::vector<int>  {1,4,2,5}, std::vector<int>  {1,4,3});
        printsome(cpures.output, 10);
    }
    printFooter();
    printHeader("Constant Memory Slide");
    {
        TestAllKernels<int, 3>(std::vector<int>  {1,1,1,1}, std::vector<int>  {1,2,3,4,5,6,7,8,9,10,11,12});//,   std::vector<int>  {3,16,23,27,22,5});
        TestAllKernels<int, 3>(std::vector<int>  {1,1,1,1}, std::vector<int>  {1,2,3,4,5,6,7,8,9,10,11,12});//,   std::vector<int>  {3,16,23,27,22,5});
        TestAllKernels<int, 3>(std::vector<int>  {1,1,1,1}, std::vector<int>  {10,11,12});//,   std::vector<int>  {3,16,23,27,22,5});

    }
    printFooter();
    printHeader("Fractional Example");
    {
        TestAllKernels(std::vector<float>{1,2,3,1}, std::vector<float>{1,0.5,1}, std::vector<float>{ 1 ,2.5, 5, 4.5, 3.5, 1});
    }
    printFooter();
    printHeader("Mask Larger than Input");
    {
        TestAllKernels(std::vector<int>  {1,2,3}, std::vector<int>  {1,1,1,1,1,1,1},   std::vector<int>  {1, 3, 6, 6, 6, 6, 6, 5, 3});
        TestAllKernels(std::vector<float>{1,4,2,5}, std::vector<float>{1,4,3},   std::vector<float>{3,16,23,27,22,5});
    }
    printFooter();
    printHeader("Larger"); 
    {
        int inputsize = 1024;
        std::vector<float> input(inputsize);//(inputsize);
        std::generate(input.begin(), input.end(), []() { static float x = 0.5; x++;return x; });
        std::vector<float> filter(150);
        std::generate(filter.begin(), filter.end(), []() { static float x = -1.; x++;return x; });
        TestAllKernels(input, filter);

    }
    printFooter();
    printHeader("Very Large"); 
    {
        int inputsize = 1024*1024*256;
        std::vector<float> input(inputsize);//(inputsize);
        std::generate(input.begin(), input.end(), []() { static float x = 0.5; x++;return x; });
        std::vector<float> filter(300);
        std::generate(filter.begin(), filter.end(), []() { static float x = -1.; x++;return x; });
        TestAllKernels(input, filter);
    }
    printFooter();
}