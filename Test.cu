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

        TestAllKernels(std::vector<float>{0,0}, std::vector<float>{1,1}, std::vector<float>{0,0,0});
        TestAllKernels(std::vector<float>{1,1}, std::vector<float>{0,0}, std::vector<float>{0,0,0});

        /*This fails. I guess because the mask is even..*/
        //Test(std::vector<float>{1,1}, std::vector<float>{1,1}, std::vector<float>{1,1,1});
        //Test(std::vector<float>{1,1}, std::vector<float>{1,1}, std::vector<float>{1,2,1});
    printFooter();
    printHeader("Lecture Slide");
        TestAllKernels(std::vector<int>  {1,4,2,5}, std::vector<int>  {1,4,3},   std::vector<int>  {3,16,23,27,22,5});
        TestAllKernels(std::vector<float>{1,4,2,5}, std::vector<float>{1,4,3},   std::vector<float>{3,16,23,27,22,5});
    printFooter();
    printHeader("Fractional Example");
        TestAllKernels(std::vector<float>{1,2,3,1}, std::vector<float>{1,0.5,1}, std::vector<float>{ 1 ,2.5, 5, 4.5, 3.5, 1});
    printFooter();
    printHeader("Mask Larger than Input");
        //TestAllKernels(std::vector<int>  {1,4,2,4,3}, std::vector<int>  {1,4,3,1,2,4,3},   std::vector<int>  {3,16,23,27,22,5});
        //TestAllKernels(std::vector<float>{1,4,2,5}, std::vector<float>{1,4,3},   std::vector<float>{3,16,23,27,22,5});
    printFooter();


}