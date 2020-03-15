#include <iostream>
#include "Test.h"
#include "Benchmark.h"

bool enableconsole = true;

void runCommand(std::string command){
    std::cout << command << std::endl;
    if (command == "test") RunTests(); else
    if (command == "benchmark") Benchmark(); else
    enableconsole = true;
}

int main(int argc, char** argv){
    for (int i = 0; i < argc; i++){
        std::string parameter = std::string(argv[i]);
        int pos = parameter.find("--");
        if (pos != std::string::npos){
            std::string token = parameter.substr(pos+2, parameter.size()); 
            runCommand(token);
        }
    }
    if (enableconsole){
        std::cout << "Please enter a command. Current options are: "; std::cout << std::endl;
        std::cout << "\ttest      - runs unit tests" << std::endl;
        std::cout << "\tbenchmark - runs benchmark mode" << std::endl;
        std::cout << "\tquit      - exits console" << std::endl;
        std::cout << std::endl;
        std::string x = "";
        do{
            std::cout << "Convolution Console: ";
            runCommand(x);
        } while ((std::cin >> x) && x != "quit");
    }

}