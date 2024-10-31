/*
* Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
* Description: remote diag doip dispatcher
*/

#include "remote_diag/include/common/remote_diag_config.h"
#include "remote_diag/include/common/remote_diag_logger.h"
#include "remote_diag/include/transport/remote_diag_doip_dispatcher.h"
#include "remote_diag/include/handler/remote_diag_handler.h"

namespace hozon {
namespace netaos {
namespace remote_diag {

const std::string REMOTE_DIAG_DOIP_CONFIG_PATH = "/app/conf/remote_diag_doip_config.json";

RemoteDiagDoipDispatcher::RemoteDiagDoipDispatcher()
: service_doip_(new DoIPTransport())
, speed_get_status_(SpeedGetStatus::Default)
{
}

RemoteDiagDoipDispatcher::~RemoteDiagDoipDispatcher()
{
}

void
RemoteDiagDoipDispatcher::Init()
{
    DGR_INFO << "RemoteDiagDoipDispatcher::Init";
    if (nullptr == service_doip_) {
        DGR_ERROR << "RemoteDiagDoipDispatcher::Init service_doip_ is nullptr.";
        return;
    }

    auto initResult = service_doip_->DoipInit(std::bind(&RemoteDiagDoipDispatcher::DoipIndicationCallback, this, std::placeholders::_1),
                                              std::bind(&RemoteDiagDoipDispatcher::DoipConfirmCallback, this, std::placeholders::_1),
                                              std::bind(&RemoteDiagDoipDispatcher::DoipRouteCallback, this, std::placeholders::_1),
                                              REMOTE_DIAG_DOIP_CONFIG_PATH);
    if (DOIP_RESULT_OK != initResult) {
        DGR_ERROR << "RemoteDiagDoipDispatcher::DoIPStart DoipInit failed. failcode: " << initResult;
    }
}

void
RemoteDiagDoipDispatcher::DeInit()
{
    DGR_INFO << "RemoteDiagDoipDispatcher::DeInit";
    if (nullptr != service_doip_) {
        service_doip_->DoipDeinit();
        delete service_doip_;
        service_doip_ = nullptr;
    }
}

void
RemoteDiagDoipDispatcher::DoipRequestByEquip(const RemoteDiagReqUdsMessage& udsMessage, const bool speedFlag)
{
    DGR_INFO << "RemoteDiagDoipDispatcher::DoipRequestByEquip speedFlag " << speedFlag
              << ", sa: " << UINT16_TO_STRING(udsMessage.udsSa)
              << ", ta: " << UINT16_TO_STRING(udsMessage.udsTa)
              << ", data: " << UINT8_VEC_TO_STRING(udsMessage.udsData);
    if (nullptr == service_doip_) {
        DGR_ERROR << "RemoteDiagDoipDispatcher::DoipRequestByEquip service_doip_ is nullptr.";
        if (speedFlag) {
            speed_get_status_ = SpeedGetStatus::Failed;
        }

        return;
    }

    doip_request_t request;
    request.logical_source_address = udsMessage.udsSa;
    request.logical_target_address = udsMessage.udsTa;
    request.ta_type = DOIP_TA_TYPE::DOIP_TA_TYPE_PHYSICAL;
    if (0xE400 == udsMessage.udsTa) {
        request.ta_type = DOIP_TA_TYPE::DOIP_TA_TYPE_FUNCTIONAL;
    }

    request.data_length = udsMessage.udsData.size();
    request.data = (char*)udsMessage.udsData.data();
    auto requestResult = service_doip_->DoipRequestByEquip(&request);
    if (DOIP_RESULT::DOIP_RESULT_OK != requestResult) {
        DGR_ERROR << "RemoteDiagDoipDispatcher::DoipRequestByEquip failed. failcode: " << UINT8_TO_STRING(requestResult);
        if (speedFlag) {
            speed_get_status_ = SpeedGetStatus::Failed;
        }

        return;
    }

    RemoteDiagHandler::getInstance()->SetRemoteRequestStatus(udsMessage.udsSa, true);
}

bool
RemoteDiagDoipDispatcher::GetVehicleSpeed()
{
    speed_get_status_ = SpeedGetStatus::Progess;
    RemoteDiagReqUdsMessage udsMessage;
    udsMessage.udsSa = 0xF000;
    udsMessage.udsTa = 0x10C3;
    udsMessage.busType = DiagUdsBusType::kServer;
    udsMessage.udsData = {0x22, 0xB1, 0x00};
    DoipRequestByEquip(udsMessage, true);
    for (uint i = 0; i < 100; i++) {
        if ((SpeedGetStatus::Failed == speed_get_status_) || (SpeedGetStatus::Success == speed_get_status_)) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    if (SpeedGetStatus::Success != speed_get_status_) {
        return false;
    }

    return true;
}

void
RemoteDiagDoipDispatcher::DoipConfirmCallback(doip_confirm_t* doipConfirm)
{
    DGR_DEBUG << "RemoteDiagDoipDispatcher::DoipConfirmCallback ta_type " << doipConfirm->ta_type
              << ", result code: " << UINT8_TO_STRING(doipConfirm->result)
              << ", sa: " << UINT16_TO_STRING(doipConfirm->logical_source_address)
              << ", ta: " << UINT16_TO_STRING(doipConfirm->logical_target_address);

    if (nullptr == doipConfirm) {
        DGR_ERROR << "RemoteDiagDoipDispatcher::DoipConfirmCallback doipConfirm is nullptr.";
        return;
    }

    if (DOIP_RESULT::DOIP_RESULT_BUSY == static_cast<uint8_t>(doipConfirm->result)) {
        DGR_INFO << "RemoteDiagDoipDispatcher::DoipConfirmCallback result doip is busy.";
    }
    else {
        if (DOIP_RESULT::DOIP_RESULT_OK != doipConfirm->result) {
            DGR_ERROR << "RemoteDiagDoipDispatcher::DoipConfirmCallback result error. code = " << UINT8_TO_STRING(doipConfirm->result);
        }
    }
}

void
RemoteDiagDoipDispatcher::DoipIndicationCallback(doip_indication_t* doipIndication)
{
    DGR_DEBUG << "RemoteDiagDoipDispatcher::DoipIndicationCallback. ta_type " << doipIndication->ta_type
              << " result code: " << UINT8_TO_STRING(doipIndication->result)
              << " sa: " << UINT16_TO_STRING(doipIndication->logical_source_address)
              << " ta: " << UINT16_TO_STRING(doipIndication->logical_target_address);
    const RemoteDiagConfigInfo& configInfo = RemoteDiagConfig::getInstance()->GetRemoteDiagConfigInfo();
    auto itr = find(configInfo.RemoteAddressList.begin(), configInfo.RemoteAddressList.end(), doipIndication->logical_target_address);
    if (itr == configInfo.RemoteAddressList.end()) {
        return;
    }

    if ((nullptr == doipIndication) || (0 == doipIndication->data_length)) {
        DGR_ERROR << "RemoteDiagDoipDispatcher::DoipIndicationCallback doipIndication is nullptr or data length is 0.";
        return;
    }

    if (DOIP_RESULT::DOIP_RESULT_OK != doipIndication->result) {
        DGR_ERROR << "RemoteDiagDoipDispatcher::DoipIndicationCallback result error. code = " << UINT8_TO_STRING(doipIndication->result);
    }

    // doip message handle
    Json::Value respMessage;
    respMessage["SA"] = UINT16_TO_STRING_DATA(doipIndication->logical_source_address);
    respMessage["TA"] = UINT16_TO_STRING_DATA(doipIndication->logical_target_address);
    respMessage["TYPE"] = REMOTE_DIAG_REQUEST_TYPE[RemoteDiagDataType::kUdsCommand];
    std::vector<uint8_t> udsData;
    udsData.assign(doipIndication->data, doipIndication->data + doipIndication->data_length);
    respMessage["DATA"] = UINT8_VEC_TO_STRING_DATA(udsData);

    DGR_INFO << "RemoteDiagDoipDispatcher::DoipIndicationCallback udsData " << UINT8_VEC_TO_STRING_DATA(udsData);
    if ((5 == udsData.size()) && (0x62 == udsData[0] && 0xB1 == udsData[1] && 0x00 == udsData[2])) {
        if (SpeedGetStatus::Progess == speed_get_status_) {
            double speed = static_cast<double>(((static_cast<uint16_t>(udsData[3]) << 8) + udsData[4]) / 100.00);
            RemoteDiagHandler::getInstance()->SetVehicleSpeed(speed);
            speed_get_status_ = SpeedGetStatus::Success;
            RemoteDiagHandler::getInstance()->SetRemoteRequestStatus(doipIndication->logical_target_address, false);
            return;
        }
    }

    RemoteDiagHandler::getInstance()->ReplyRemoteMessage(respMessage);
    RemoteDiagHandler::getInstance()->SetRemoteRequestStatus(doipIndication->logical_target_address, false);
}

}  // namespace remote_diag
}  // namespace netaos
}  // namespace hozon