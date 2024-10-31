
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include "get_ifdata.h"

namespace hozon {
namespace netaos {
namespace tools {


void
IfDataInfo::PrintUsage()
{
    std::cout << "Usage: nos devm ifdata [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  si <interface>,     Used to Print all statistics on input." << std::endl;
    std::cout << "  bips <interface>,   Used to Print # of incoming bytes per second." << std::endl;
    std::cout << "  bops <interface>,   Used to Print # of outgoing bytes per second." << std::endl;
    std::cout << "  help,               Used to show the Usage." << std::endl;
    std::cout << std::endl;
}
int32_t
IfDataInfo::StartGetIfdata()
{
    if (arguments_.size() < 2) {
        PrintUsage();
        return -1;
    }
    std::string cmd;

    if (arguments_[0] == "si") {
        std::cout << "si." << std::endl;
        cmd.clear();
        cmd += "ifdata -si ";
        cmd += arguments_[1];
        while(true) {
            if (system(cmd.c_str()) < 0) {
                std::cout << "system cmd err" << std::endl;
                return -1;
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    else if (arguments_[0] == "bips") {
        std::cout << "bips." << std::endl;
        cmd.clear();
        cmd += "ifdata -bips ";
        cmd += arguments_[1];
        cmd += "&";
        while(true) {
            if (system(cmd.c_str()) < 0) {
                std::cout << "system cmd err" << std::endl;
                return -1;
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    else if (arguments_[0] == "bops") {
        std::cout << "bops." << std::endl;
        cmd.clear();
        cmd += "ifdata -bops ";
        cmd += arguments_[1];
        cmd += "&";
        while(true) {
            if (system(cmd.c_str()) < 0) {
                return -1;
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    else {
        std::cout << "Invalid value" << std::endl;
    }

    return 0;
}

}
}
}
