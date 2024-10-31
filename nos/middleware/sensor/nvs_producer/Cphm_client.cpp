/*
* Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
* Description: phm client instance
*/

#include "Cphm_client.h"
#include "CUtils.hpp"

CameraPhmClient* CameraPhmClient::instance_ = nullptr;
std::mutex CameraPhmClient::mtx_;

CameraPhmClient::CameraPhmClient() : phm_client_ptr_(new PHMClient()) {}

CameraPhmClient* CameraPhmClient::getInstance() {
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new CameraPhmClient();
        }
    }

    return instance_;
}

void CameraPhmClient::Init() {
    if (nullptr != phm_client_ptr_) {
        phm_client_ptr_->Init("", std::bind(&CameraPhmClient::CameraServiceAvailableCallback, this, std::placeholders::_1),
                              std::bind(&CameraPhmClient::CameraFaultReceiveCallback, this, std::placeholders::_1));
    }
}

void CameraPhmClient::DeInit() {
    if (nullptr != phm_client_ptr_) {
        phm_client_ptr_->Deinit();
        delete phm_client_ptr_;
        phm_client_ptr_ = nullptr;
    }

    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
}

int32_t CameraPhmClient::CameraReportFault(const SendFault_t& faultInfo) {
    if (nullptr == phm_client_ptr_) {
        LOG_ERR("CameraPhmClient::CameraReportFault error: phm_client_ptr_ is nullptr!\n");
        return -1;
    }

    int32_t result = phm_client_ptr_->ReportFault(faultInfo);
    if (0 != result) {
        LOG_ERR("CameraPhmClient::CameraReportFault error result:%d\n", result);
    }
    return result;
}

void CameraPhmClient::CameraServiceAvailableCallback(const bool bResult) {
    LOG_INFO("CameraPhmClient::CameraServiceAvailableCallback bResult:%d\n", bResult);
}

void CameraPhmClient::CameraFaultReceiveCallback(const ReceiveFault_t& fault) {
    LOG_INFO("CameraPhmClient::CameraFaultReceiveCallback faultId:%d faultObj:%d faultStatus:%d faultOccurTime:%d faultDomain:%d faultDes:%d\n",
            fault.faultId, static_cast<uint>(fault.faultObj), fault.faultStatus, fault.faultOccurTime, fault.faultDomain, fault.faultDes);

    for (auto& item : fault.faultCluster) {
        LOG_INFO("CameraPhmClient::CameraFaultReceiveCallback:%d clusterLevel:%d\n", item.clusterName, item.clusterValue);
    }
}
