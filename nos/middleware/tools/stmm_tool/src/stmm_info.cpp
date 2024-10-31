#include <iostream>
#include "system_monitor_info.h"

int main(int argc, char * argv[])
{
    SystemMonitorInfo::getInstance()->Init();
    std::string info = "";
    if (2 != argc) {
        for (uint i = 0; i < MONITOR_INFO_ORDER.size(); i++) {
            info = SystemMonitorInfo::getInstance()->GetMonitorInfo(MONITOR_INFO_ORDER[i]);
            if ("" == info) {
                info = SystemMonitorInfo::getInstance()->GetMonitorInfoFromFile(MONITOR_INFO_ORDER[i]);
            }

            if ("" != info) {
                std::cout << info;
            }
        }
    }
    else {
        std::string type = std::string(argv[1]);
        if ("system" == type) {
            for (uint i = 0; i < SYSTEM_MONITOR_INFO_ORDER.size(); i++) {
                info = SystemMonitorInfo::getInstance()->GetMonitorInfo(SYSTEM_MONITOR_INFO_ORDER[i]);
                if ("" == info) {
                    info = SystemMonitorInfo::getInstance()->GetMonitorInfoFromFile(SYSTEM_MONITOR_INFO_ORDER[i]);
                }

                if ("" != info) {
                    std::cout << info;
                }
            }
        }
        else if ("device" == type) {
            for (uint i = 0; i < DEVICE_MONITOR_INFO_ORDER.size(); i++) {
                info = SystemMonitorInfo::getInstance()->GetMonitorInfo(DEVICE_MONITOR_INFO_ORDER[i]);
                if ("" == info) {
                    info = SystemMonitorInfo::getInstance()->GetMonitorInfoFromFile(DEVICE_MONITOR_INFO_ORDER[i]);
                }

                if ("" != info) {
                    std::cout << info;
                }
            }
        }
        else if ("safety" == type) {
            for (uint i = 0; i < SAFETY_MONITOR_INFO_ORDER.size(); i++) {
                info = SystemMonitorInfo::getInstance()->GetMonitorInfo(SAFETY_MONITOR_INFO_ORDER[i]);
                if ("" == info) {
                    info = SystemMonitorInfo::getInstance()->GetMonitorInfoFromFile(SAFETY_MONITOR_INFO_ORDER[i]);
                }

                if ("" != info) {
                    std::cout << info;
                }
            }
        }
        else {
            info = SystemMonitorInfo::getInstance()->GetMonitorInfo(type);
            if ("" == info) {
                info = SystemMonitorInfo::getInstance()->GetMonitorInfoFromFile(type);
            }

            if ("" != info) {
                std::cout << info;
            }
        }
    }

    SystemMonitorInfo::getInstance()->DeInit();
    return 0;
}