#include "Convolution.h"
#include "Test.h"

void RunTests(){
    /*Testing Convolution Example from Slide as an Integr and a float*/
    /*Input, Filter, Expected Value*/
    Test(std::vector<int>  {1,4,2,5}, std::vector<int>  {1,4,3},   std::vector<int>  {3,16,23,27,22,5});
    Test(std::vector<float>{1,4,2,5}, std::vector<float>{1,4,3},   std::vector<float>{3,16,23,27,22,5});
    Test(std::vector<float>{1,2,3,1}, std::vector<float>{1,0.5,1}, std::vector<float>{ 1 ,2.5, 5, 4.5, 3.5, 1});


}