#include <iostream>
#include <thread>
#include <chrono>
#include <signal.h>
#include <stdint.h>
#include <memory>

#include "ota_api_sample.h"
#include "log/include/logging.h"

using namespace std;

uint8_t stopFlag = 0;
OTAApiSample ota;


void SigHandler(int signum)
{
    std::cout << "--- ota_api_sample sigHandler enter, signum [" << signum << "] ---" << std::endl;
    stopFlag = 1;
    ota.DeInit();
}


int main(int argc, char* argv[])
{
    signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);

    if (argc < 2) {
        std::cout << "please input upgrade file path!" << std::endl;
        return -1;
    }

    std::string path = argv[1];

    ota.Init();
    int res = ota.Run(path);
    if (res < 0) {
        stopFlag = 1;
        ota.DeInit();
    }

    return 0;
}