#include <iostream>
#include "Test.h"
#include "Benchmark.h"

bool enableconsole = true;

void Help(){
    std::cout << "Please enter a command. Current options are: "; std::cout << std::endl;
    std::cout << "\thelp      - dispays this prompt" << std::endl;
    std::cout << "\ttest      - runs unit tests" << std::endl;
    std::cout << "\tbenchmark - runs benchmark mode" << std::endl;
    std::cout << "\tquit      - exits console" << std::endl;
    std::cout << std::endl;
}

void runCommand(std::string command){
    if (command == "test") { RunTests(); enableconsole = false; }
    if (command == "benchmark") {  Benchmark(); enableconsole = false; }
    if (command == "help") { Help(); }
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
        Help();
        std::string x = "";
        do{
            runCommand(x);
            std::cout << std::endl << "Convolution Console: ";
        } while ((std::cin >> x) && x != "quit");
    }

}