#include "Convolution.h"
#include "Test.h"

int main(int argc, char** argv){
    for (int i = 0; i < argc; i++){
        std::string parameter = std::string(argv[i]);
        if (parameter == "--test") RunTests();
    }
    RunTests();
}