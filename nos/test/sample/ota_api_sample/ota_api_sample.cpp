#include <cstdint>
#include <cstdio>
#include <memory>
#include <thread>
#include <iostream>
#include <string>

#include "ota_api_sample.h"

OTAApiSample::OTAApiSample() : stop_flag_(0) {
    std::cout << "OTAApiSample::OTAApiSample()" << std::endl;
}

OTAApiSample::~OTAApiSample() {
    std::cout << "OTAApiSample::~OTAApiSample()" << std::endl;
}

void OTAApiSample::Init() {
    std::cout << "OTAApiSample::Init()" << std::endl;
    STD_RTYPE_E res = ota_api_init();
    std::cout << "OTAApiSample::ota_api_init() res: " << res << std::endl;
}

void OTAApiSample::DeInit() {
    std::cout << "OTAApiSample::DeInit()" << std::endl;
    stop_flag_ = 2;
    STD_RTYPE_E res = ota_api_deinit();
    std::cout << "OTAApiSample::ota_api_deinit() res: " << res << std::endl;
}

int OTAApiSample::Run(std::string upgrade_file_path) {
    std::cout << "OTAApiSample::Run(" << upgrade_file_path << ")" << std::endl;

    int8_t version[10] = {'\0'};
    STD_RTYPE_E res = ota_get_api_version(version);
    if (res == E_OK) {
        std::string ss((char*)version);
        std::cout << "api_version: " << ss << std::endl;
    } else {
        std::cout << "ota_get_api_version failed! " << std::endl;
        return -1;
    }

    enum OTA_UPDATE_STATUS_E ota_update_status = OTA_UPDATE_STATUS_IDLE; 
    uint8_t progress = 0;

    do {
        res = ota_get_update_status(&ota_update_status, &progress);
        if (res == E_OK) {
            if (ota_update_status == OTA_UPDATE_STATUS_IDLE) {
                res = ota_start_update((uint8_t*)upgrade_file_path.c_str());
                if (res != E_OK) {
                    std::cout << "ota_start_update failed! " << std::endl;
                    return -1;
                }

                while (!stop_flag_) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                    ota_get_update_status(&ota_update_status, &progress);
                    printf("update_status[%d], progress[%d %%]\n", ota_update_status, progress);
                    if (90 == progress) {
                        stop_flag_ = 1;
                        break;
                    }
                }
            } else {
                std::cout << "ota_update_status not IDLE, sleep 2s to retry!" << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(2000));
                continue;
            }
        } else {
            std::cout << "ota_get_update_status failed! " << std::endl;
            return -1;
        }
    } while(!stop_flag_);

    if (1 == stop_flag_) {
        std::cout << "update success!!! " << std::endl;
        ota_api_deinit();
    } else if (2 == stop_flag_) {
        std::cout << "update aborted!!! " << std::endl;
    }

    return 0;
}