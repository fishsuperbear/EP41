/*
* Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
* Description: tpl method sender
*/

#include "diag_server_transport_method_sender.h"
#include "diag/diag_server/include/common/diag_server_logger.h"
#include "diag_server_transport_cm.h"
#include "diag/diag_server/include/common/diag_server_config.h"
#include "diag/diag_server/include/info/diag_server_chassis_info.h"

namespace hozon {
namespace netaos {
namespace diag {
namespace cm_transport {


const uint32_t REQUEST_TIMEOUT = 0xFFFFFFFF;

DiagServerTransportMethodSender::DiagServerTransportMethodSender()
: update_status_client_(nullptr)
{
}

DiagServerTransportMethodSender::~DiagServerTransportMethodSender()
{
}

void
DiagServerTransportMethodSender::Init()
{
    DG_INFO << "DiagServerTransportMethodSender::Init";
    std::shared_ptr<uds_data_methodPubSubType> req_data_type = std::make_shared<uds_data_methodPubSubType>();
    std::shared_ptr<uds_data_methodPubSubType> resp_data_type = std::make_shared<uds_data_methodPubSubType>();
    std::vector<std::string> service;
    bool bResult = DiagServerConfig::getInstance()->QueryAllExternalService(service);
    if (bResult) {
        for (auto& item: service) {
            auto client = std::make_shared<Client<uds_data_method, uds_data_method>>(req_data_type, resp_data_type);
            client->Init(0, item);
            client_map_.insert(std::make_pair(item, client));
        }
    }

    std::shared_ptr<ChassisOtaMethodPubSubType> req_chassis_type = std::make_shared<ChassisOtaMethodPubSubType>();
    std::shared_ptr<ChassisOtaMethodPubSubType> resp_chassis_type = std::make_shared<ChassisOtaMethodPubSubType>();
    chassis_info_client_ = std::make_shared<Client<ChassisOtaMethod, ChassisOtaMethod>>(req_chassis_type, resp_chassis_type);
    chassis_info_client_->Init(0, "/soc/chassis_ota_method");

    std::shared_ptr<update_status_methodPubSubType> req_status_type = std::make_shared<update_status_methodPubSubType>();
    std::shared_ptr<update_status_methodPubSubType> resp_status_type = std::make_shared<update_status_methodPubSubType>();
    update_status_client_ = std::make_shared<Client<update_status_method, update_status_method>>(req_status_type, resp_status_type);
    update_status_client_->Init(0, "update_status");
}

void
DiagServerTransportMethodSender::DeInit()
{
    DG_INFO << "DiagServerTransportMethodSender::DeInit";
    if (nullptr != update_status_client_) {
        update_status_client_->Deinit();
        update_status_client_ = nullptr;
    }

    if (nullptr != chassis_info_client_) {
        chassis_info_client_->Deinit();
        chassis_info_client_ = nullptr;
    }

    for (auto& item: client_map_) {
        if (nullptr != item.second) {
            item.second->Deinit();
            item.second = nullptr;
        }
    }
}

void
DiagServerTransportMethodSender::DiagMethodSend(const uint8_t sid, const uint8_t subFunc, const std::vector<std::string> service, std::vector<uint8_t>& udsData)
{
    DG_DEBUG << "DiagServerTransportMethodSender::DiagMethodSend sid: " << UINT8_TO_STRING(sid) << " subFunc: " << UINT8_TO_STRING(subFunc)
                                                   << " service.size: " << service.size() << " udsdata.size: " << udsData.size() <<  " udsdata: " << UINT8_VEC_TO_STRING(udsData);
    if (client_map_.empty()) {
        DG_ERROR << "DiagServerTransportMethodSender::DiagMethodSend client map is empty.";
        return;
    }

    std::shared_ptr<uds_data_method> req_data = std::make_shared<uds_data_method>();
    req_data->sid(sid);
    req_data->subid(subFunc);
    std::vector<uint8_t> dataVec;
    dataVec.assign(udsData.begin(), udsData.end());
    req_data->data_len(dataVec.size());
    req_data->data_vec(dataVec);

    // Request
    req_data->fire_forget(false);
    udsData.clear();
    for (uint i = 0; i < service.size(); i++) {
        DG_DEBUG << "DiagServerTransPortCM::DiagMethodSend service name: " << service[i];
        auto itr = client_map_.find(service[i]);
        if (itr == client_map_.end()) {
            DG_WARN << "DiagServerTransportMethodSender::DiagMethodSend error service name: " << service[i];
            continue;
        }

        if (nullptr == itr->second) {
            DG_ERROR << "DiagServerTransportMethodSender::DiagMethodSend service name: " << service[i] << "client is nullptr.";
            continue;
        }

        // if cm method not connect, wait connect(timeout: 5s)
        for (int i  = 0; i < 100; i++) {
            if (0 == itr->second->WaitServiceOnline(0)) {
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        std::shared_ptr<uds_data_method> resq_data = std::make_shared<uds_data_method>();
        int iResult = itr->second->Request(req_data, resq_data, REQUEST_TIMEOUT);
        if (0 != iResult) {
            DG_ERROR << "DiagServerTransportMethodSender::DiagMethodSend service name: " << service[i] << " request failed.";
            udsData.push_back(DiagServerNrcErrc::kNegativeHead);
            udsData.push_back(sid);
            udsData.push_back(DiagServerNrcErrc::kConditionsNotCorrect);
            return;
        }

        if (resq_data->resp_ack()) {
            udsData.push_back(DiagServerNrcErrc::kNegativeHead);
            udsData.push_back(resq_data->sid());

            for (auto& item : resq_data->data_vec()) {
                udsData.push_back(item);
            }

            return;
        }

        if (i == (service.size() - 1)) {
            if (resq_data->resp_ack()) {
                udsData.push_back(DiagServerNrcErrc::kNegativeHead);
                udsData.push_back(resq_data->sid());
            }
            else {
                udsData.push_back(resq_data->sid() + 0x40);
                if (0xFF != resq_data->subid()) {
                    udsData.push_back(resq_data->subid());
                }
            }

            for (auto& item : resq_data->data_vec()) {
                udsData.push_back(item);
            }

            return;
        }
    }
}

void
DiagServerTransportMethodSender::ChassisMethodSend()
{
    DG_DEBUG << "DiagServerTransportMethodSender::ChassisMethodSend.";
    if (nullptr == chassis_info_client_) {
        DG_ERROR << "DiagServerTransportMethodSender::ChassisMethodSend chassis_info_client_ is nullptr.";
        return;
    }

    // if cm method not connect, wait connect(timeout: 5s)
    for (int i  = 0; i < 100; i++) {
        if (0 == chassis_info_client_->WaitServiceOnline(0)) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::shared_ptr<ChassisOtaMethod> req_chassis = std::make_shared<ChassisOtaMethod>();
    std::shared_ptr<ChassisOtaMethod> resq_chassis = std::make_shared<ChassisOtaMethod>();
    int iResult = chassis_info_client_->Request(req_chassis, resq_chassis, 50);
    if (0 != iResult) {
        DG_ERROR << "DiagServerTransportMethodSender::ChassisMethodSend request failed.";
        return;
    }

    DG_INFO << "DiagServerTransportMethodSender::ChassisMethodSend VCU_ActGearPosition: " << resq_chassis->VCU_ActGearPosition()
                                                              << " BDCS10_AC_OutsideTemp: " << resq_chassis->BDCS10_AC_OutsideTemp()
                                                              << " ICU2_Odometer: " << resq_chassis->ICU2_Odometer()
                                                              << " BDCS1_PowerManageMode: " << resq_chassis->BDCS1_PowerManageMode()
                                                              << " Ignition_status: " << resq_chassis->Ignition_status()
                                                              << " ESC_VehicleSpeedValid: " << resq_chassis->ESC_VehicleSpeedValid()
                                                              << " ESC_VehicleSpeed: " << resq_chassis->ESC_VehicleSpeed();
    DiagServerChassisInfo::getInstance()->SetGearDisplay(resq_chassis->VCU_ActGearPosition());
    DiagServerChassisInfo::getInstance()->SetOutsideTemp(resq_chassis->BDCS10_AC_OutsideTemp());
    DiagServerChassisInfo::getInstance()->SetOdometer(resq_chassis->ICU2_Odometer());
    DiagServerChassisInfo::getInstance()->SetPowerMode(resq_chassis->BDCS1_PowerManageMode());
    DiagServerChassisInfo::getInstance()->SetIgStatus(resq_chassis->Ignition_status());
    DiagServerChassisInfo::getInstance()->SetVehicleSpeedValid(resq_chassis->ESC_VehicleSpeedValid());
    DiagServerChassisInfo::getInstance()->SetVehicleSpeed(resq_chassis->ESC_VehicleSpeed());
}

bool
DiagServerTransportMethodSender::IsCheckUpdateStatusOk()
{
    DG_DEBUG << "DiagServerTransportMethodSender::IsCheckUpdateStatusOk.";
    if (nullptr == update_status_client_) {
        DG_ERROR << "DiagServerTransportMethodSender::IsCheckUpdateStatusOk update_status_client_ is nullptr.";
        return false;
    }

    std::shared_ptr<update_status_method> req_status = std::make_shared<update_status_method>();
    req_status->update_status(DiagUpdateStatus::kDefault);

    // if cm method not connect, wait connect(timeout: 5s)
    for (int i  = 0; i < 100; i++) {
        if (0 == update_status_client_->WaitServiceOnline(0)) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::shared_ptr<update_status_method> resq_status = std::make_shared<update_status_method>();
    int iResult = update_status_client_->Request(req_status, resq_status, 50);
    if (0 != iResult) {
        DG_ERROR << "DiagServerTransportMethodSender::IsCheckUpdateStatusOk request failed.";
        return false;
    }

    DG_INFO << "DiagServerTransportMethodSender::IsCheckUpdateStatusOk update_status: " << resq_status->update_status();
    if (DiagUpdateStatus::kUpdating == resq_status->update_status()) {
        return false;
    }

    return true;
}


}  // namespace cm_transport
}  // namespace diag
}  // namespace netaos
}  // namespace hozon