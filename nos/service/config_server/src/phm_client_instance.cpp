/*
 * @Author: Z04975 liguoqiang@hozonauto.com
 * @Date: 2023-08-21 15:38:46
 * @LastEditors: Z04975 liguoqiang@hozonauto.com
 * @LastEditTime: 2023-09-13 17:14:40
 * @FilePath: /nos/service/config_server/src/phm_client_instance.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 * Description: phm client instance
 */

#include "include/phm_client_instance.h"
using namespace hozon::netaos::phm;
PhmClientInstance* PhmClientInstance::instance_ = nullptr;
std::mutex PhmClientInstance::mtx_;

PhmClientInstance::PhmClientInstance() : phm_client_ptr_(new PHMClient()) {}

PhmClientInstance* PhmClientInstance::getInstance() {
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new PhmClientInstance();
        }
    }
    return instance_;
}

void PhmClientInstance::Init() {
    if (nullptr != phm_client_ptr_) {
        phm_client_ptr_->Init();
        phm_client_ptr_->Start();
    }
}

void PhmClientInstance::DeInit() {
    if (nullptr != phm_client_ptr_) {
        phm_client_ptr_->Stop();
        phm_client_ptr_->Deinit();
        delete phm_client_ptr_;
        phm_client_ptr_ = nullptr;
    }

    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
}

int32_t PhmClientInstance::ReportFault(const SendFault_t& faultInfo) {
    if (nullptr == phm_client_ptr_) {
        CONFIG_LOG_WARN << "PhmClientInstance::ReportFault error: phm_client_ptr_ is nullptr!";
        return -1;
    }
    int32_t result = phm_client_ptr_->ReportFault(faultInfo);
    return result;
}

int32_t PhmClientInstance::ReportCheckPoint(uint32_t checkPointId) {
    if (nullptr == phm_client_ptr_) {
        CONFIG_LOG_WARN << "PhmClientInstance::ReportCheckPoint error: phm_client_ptr_ is nullptr!";
        return -1;
    }
    int32_t result = phm_client_ptr_->ReportCheckPoint(checkPointId);
    return result;
}

void PhmClientInstance::InhibitFault(const std::vector<uint32_t>& faultKeys) {
    if (nullptr == phm_client_ptr_) {
        CONFIG_LOG_WARN << "PhmClientInstance::InhibitFault error: phm_client_ptr_ is nullptr!";
        return;
    }
    phm_client_ptr_->InhibitFault(faultKeys);
}

void PhmClientInstance::RecoverInhibitFault(const std::vector<uint32_t>& faultKeys) {
    if (nullptr == phm_client_ptr_) {
        CONFIG_LOG_WARN << "PhmClientInstance::RecoverInhibitFault error: phm_client_ptr_ is nullptr!";
        return;
    }
    phm_client_ptr_->RecoverInhibitFault(faultKeys);
}

void PhmClientInstance::InhibitAllFault() {
    if (nullptr == phm_client_ptr_) {
        CONFIG_LOG_WARN << "PhmClientInstance::InhibitAllFault error: phm_client_ptr_ is nullptr!";
        return;
    }
    phm_client_ptr_->InhibitAllFault();
}

void PhmClientInstance::RecoverInhibitAllFault() {
    if (nullptr == phm_client_ptr_) {
        CONFIG_LOG_WARN << "PhmClientInstance::RecoverInhibitAllFault error: phm_client_ptr_ is nullptr!";
        return;
    }
    phm_client_ptr_->RecoverInhibitAllFault();
}
