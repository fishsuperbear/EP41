/*
* Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
* Description: phm client instance
*/

#include "phm_client_instance.h"

PhmClientInstance* PhmClientInstance::instance_ = nullptr;
std::mutex PhmClientInstance::mtx_;

const std::string PHM_CONFIG_FILE_PATH = "../conf/phm_config.yaml";

PhmClientInstance::PhmClientInstance()
: phm_client_ptr_(new PHMClient())
{
}

PhmClientInstance*
PhmClientInstance::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new PhmClientInstance();
        }
    }

    return instance_;
}

void
PhmClientInstance::Init()
{
    if (nullptr != phm_client_ptr_) {
        phm_client_ptr_->Init(PHM_CONFIG_FILE_PATH, std::bind(&PhmClientInstance::ServiceAvailableCallback, this, std::placeholders::_1), std::bind(&PhmClientInstance::FaultReceiveCallback, this, std::placeholders::_1));
        phm_client_ptr_->Start();
    }
}

void
PhmClientInstance::DeInit()
{
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

int32_t
PhmClientInstance::ReportFault(const SendFault_t& faultInfo)
{
    if (nullptr == phm_client_ptr_) {
        std::cout << "PhmClientInstance::ReportFault error: phm_client_ptr_ is nullptr!" << std::endl;
        return -1;
    }

    int32_t result = phm_client_ptr_->ReportFault(faultInfo);
    return result;
}

int32_t
PhmClientInstance::ReportCheckPoint(uint32_t checkPointId)
{
    if (nullptr == phm_client_ptr_) {
        std::cout << "PhmClientInstance::ReportCheckPoint error: phm_client_ptr_ is nullptr!" << std::endl;
        return -1;
    }

    int32_t result = phm_client_ptr_->ReportCheckPoint(checkPointId);
    return result;
}

void
PhmClientInstance::InhibitFault(const std::vector<uint32_t>& faultKeys)
{
    if (nullptr == phm_client_ptr_) {
        std::cout << "PhmClientInstance::InhibitFault error: phm_client_ptr_ is nullptr!" << std::endl;
        return;
    }

    phm_client_ptr_->InhibitFault(faultKeys);
}

void
PhmClientInstance::RecoverInhibitFault(const std::vector<uint32_t>& faultKeys)
{
    if (nullptr == phm_client_ptr_) {
        std::cout << "PhmClientInstance::RecoverInhibitFault error: phm_client_ptr_ is nullptr!" << std::endl;
        return;
    }

    phm_client_ptr_->RecoverInhibitFault(faultKeys);
}

void
PhmClientInstance::InhibitAllFault()
{
    if (nullptr == phm_client_ptr_) {
        std::cout << "PhmClientInstance::InhibitAllFault error: phm_client_ptr_ is nullptr!" << std::endl;
        return;
    }

    phm_client_ptr_->InhibitAllFault();
}

void
PhmClientInstance::RecoverInhibitAllFault()
{
    if (nullptr == phm_client_ptr_) {
        std::cout << "PhmClientInstance::RecoverInhibitAllFault error: phm_client_ptr_ is nullptr!" << std::endl;
        return;
    }

    phm_client_ptr_->RecoverInhibitAllFault();
}

void
PhmClientInstance::ServiceAvailableCallback(const bool bResult)
{
    std::cout << "PhmClientInstance::ServiceAvailableCallback bResult: " << bResult << std::endl;
}

void
PhmClientInstance::FaultReceiveCallback(const ReceiveFault_t& fault)
{
    std::cout << "PhmClientInstance::FaultReceiveCallback faultId: " << fault.faultId << " faultObj: " << static_cast<uint>(fault.faultObj)
                                    << " faultStatus: " << fault.faultStatus << " faultOccurTime: " << fault.faultOccurTime
                                    << " faultDomain: " << fault.faultDomain << " faultDes: " << fault.faultDes << std::endl;

    for (auto& item : fault.faultCluster) {
        std::cout << "PhmClientInstance::FaultReceiveCallback: " << item.clusterName << " clusterValue: " << item.clusterValue << std::endl;
    }
}
