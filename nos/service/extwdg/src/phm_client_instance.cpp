/*
* Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
* Description: phm client instance
*/

#include "phm_client_instance.h"
#include "extwdg_check.h"

PhmClientInstance* PhmClientInstance::instance_ = nullptr;
std::mutex PhmClientInstance::mtx_;

const std::string PHM_CONFIG_FILE_PATH = "/app/conf/extwdg_config.yaml";

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
PhmClientInstance::ReportCheckPoint(uint32_t checkPointId)
{
    if (nullptr == phm_client_ptr_) {
        std::cout << "PhmClientInstance::ReportCheckPoint error: phm_client_ptr_ is nullptr!" << std::endl;
        return -1;
    }

    int32_t result = phm_client_ptr_->ReportCheckPoint(checkPointId);
    return result;
}

int32_t
PhmClientInstance::ReportFault(const SendFault_t& faultInfo)
{
    if (nullptr == phm_client_ptr_) {
        std::cout << "PhmClientInstance::ReportCheckPoint error: phm_client_ptr_ is nullptr!" << std::endl;
        return -1;
    }
    std::cout << "PhmClientInstance::ReportFault      faultInfo id is   " << static_cast<uint32_t>(faultInfo.faultObj)  << std::endl;
    int32_t result = phm_client_ptr_->ReportFault(faultInfo);
    return result;
}

void
PhmClientInstance::ServiceAvailableCallback(const bool bResult)
{
    std::cout << "PhmClientInstance::ServiceAvailableCallback bResult: " << bResult << std::endl;
}

int32_t
PhmClientInstance::Start()
{
    if (nullptr == phm_client_ptr_) {
        std::cout << "PhmClientInstance::Start error: phm_client_ptr_ is nullptr!" << std::endl;
        return -1;
    }

    phm_client_ptr_->Start();
    return 0;
}

void
PhmClientInstance::Stop()
{
    if (nullptr == phm_client_ptr_) {
        std::cout << "PhmClientInstance::stop error: phm_client_ptr_ is nullptr!" << std::endl;
        return;
    }

    phm_client_ptr_->Stop();
    return;
}

void
PhmClientInstance::FaultReceiveCallback(const ReceiveFault_t& fault)
{
    std::cout << "PhmClientInstance::FaultReceiveCallback faultId: " << fault.faultId << " faultObj: " << static_cast<uint>(fault.faultObj)
                                    << " faultStatus: " << fault.faultStatus << " faultOccurTime: " << fault.faultOccurTime
                                    << " faultDomain: " << fault.faultDomain << " faultDes: " << fault.faultDes << std::endl;
    std::cout << "set and check condition is " << hozon::netaos::extwdg::SetFlag::getInstance()->GetCheckCase() <<std::endl;
    for (auto& item : fault.faultCluster) {
        std::cout << "PhmClientInstance::FaultReceiveCallback: " << item.clusterName << " clusterValue: " << item.clusterValue << std::endl;
    }
    if(hozon::netaos::extwdg::SetFlag::getInstance()->GetCheckCase() == "alive" && fault.faultId == 4920 && fault.faultObj == 1) {
        std::cout << "Alivetest: "<< " faultId: " << fault.faultId << " faultObj: " << static_cast<uint>(fault.faultObj) << std::endl;
        if(fault.faultStatus == 1) {
            std::cout << "Alivetest: "<< " faultStatus: " << static_cast<uint>(fault.faultStatus) << std::endl;
            // hozon::netaos::extwdg::SetReportFlag();
            hozon::netaos::extwdg::SetFlag::getInstance()->SetReportFlag();
        } 
        else if (fault.faultStatus == 0) {
            std::cout << "Alivetest: "<< " faultStatus: " << static_cast<uint>(fault.faultStatus) << std::endl;
            hozon::netaos::extwdg::SetFlag::getInstance()->SetRecoverFlag();
        }
        return;
    } 
    else if (hozon::netaos::extwdg::SetFlag::getInstance()->GetCheckCase() == "deadline"&& fault.faultId == 4920 && fault.faultObj == 2) {
        std::cout << "Deadlinetest: "<< " faultId: " << fault.faultId << " faultObj: " << static_cast<uint>(fault.faultObj) << std::endl;
        if(fault.faultStatus == 1) {
            std::cout << "Deadlinetest: "<< " faultStatus: " << static_cast<uint>(fault.faultStatus) << std::endl;
            hozon::netaos::extwdg::SetFlag::getInstance()->SetReportFlag();
        } 
        else if (fault.faultStatus == 0) {
            std::cout << "Deadlinetest: "<< " faultStatus: " << static_cast<uint>(fault.faultStatus) << std::endl;
            hozon::netaos::extwdg::SetFlag::getInstance()->SetRecoverFlag();
        }
        return;
    }
    else if (hozon::netaos::extwdg::SetFlag::getInstance()->GetCheckCase() == "logic"&& fault.faultId == 4920 && fault.faultObj == 3) {
        std::cout << "Logictest: "<< " faultId: " << fault.faultId << " faultObj: " << static_cast<uint>(fault.faultObj) << std::endl;
        if(fault.faultStatus == 1) {
            std::cout << "Logictest: "<< " faultStatus: " << static_cast<uint>(fault.faultStatus) << std::endl;
            hozon::netaos::extwdg::SetFlag::getInstance()->SetReportFlag();
        } 
        else if (fault.faultStatus == 0) {
            std::cout << "Logictest: "<< " faultStatus: " << static_cast<uint>(fault.faultStatus) << std::endl;
            hozon::netaos::extwdg::SetFlag::getInstance()->SetRecoverFlag();
        }
        return;
    }
    
}
