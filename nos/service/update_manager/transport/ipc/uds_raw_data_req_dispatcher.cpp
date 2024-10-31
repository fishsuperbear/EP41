/*
* Copyright (c) Hozon Auto Co., Ltd. 2023-2023. All rights reserved.
* Description: uds raw data event sender
*/

#include "uds_raw_data_req_dispatcher.h"
#include "update_manager/log/update_manager_logger.h"

namespace hozon {
namespace netaos {
namespace update {


UdsRawDataReqDispatcher::UdsRawDataReqDispatcher()
    :  client_(nullptr)
{
}

UdsRawDataReqDispatcher::~UdsRawDataReqDispatcher()
{
}

void
UdsRawDataReqDispatcher::Init()
{
    UPDATE_LOG_D("Init");
    client_ = std::make_shared<IPCClient>();
    client_->Init("uds_raw_data_req_eventTopic");
}

void
UdsRawDataReqDispatcher::Deinit()
{
    UPDATE_LOG_D("Deinit");
    if (client_ != nullptr) {
        client_->Deinit();
        client_ = nullptr;
    }
}

void
UdsRawDataReqDispatcher::Send(UdsRawDataReqEvent& sendUdsRawDataReq)
{
    UdsRawDataReqMethod reqProtoData{};
    reqProtoData.set_sa(sendUdsRawDataReq.sa);
    reqProtoData.set_ta(sendUdsRawDataReq.ta);
    reqProtoData.set_bus_type(sendUdsRawDataReq.bus_type);
    reqProtoData.set_data_vec(sendUdsRawDataReq.data_vec.data(), sendUdsRawDataReq.data_vec.size());
    std::vector<uint8_t> vecProto = std::vector<uint8_t>(reqProtoData.data_vec().begin(), reqProtoData.data_vec().end());
    
    if (client_ == nullptr) {
        UPDATE_LOG_E("client_ is nullptr!.");
        return;
    }

    if (client_->IsMatched() != 0) {
        UPDATE_LOG_E("client_ is not matched!.");
        return;
    }

    std::string req = reqProtoData.SerializeAsString();
    std::vector<uint8_t> data;
    data.assign(req.begin(), req.end());

    if (client_->RequestAndForget(data) != 0) {
        UPDATE_LOG_E("UdsRawDataReqDispatcher failed Send --> ReqSa: %X, reqTa: %X, updateType: %d, uds data size: %ld, data: [%s].",
        reqProtoData.sa(), reqProtoData.ta(), reqProtoData.bus_type(), reqProtoData.data_vec().size(), (UM_UINT8_VEC_TO_HEX_STRING(vecProto)).c_str());
    }
    else {
        UPDATE_LOG_D("UdsRawDataReqDispatcher success Send --> ReqSa: %X, reqTa: %X, updateType: %d, uds data size: %ld, data: [%s].",
        reqProtoData.sa(), reqProtoData.ta(), reqProtoData.bus_type(), reqProtoData.data_vec().size(), (UM_UINT8_VEC_TO_HEX_STRING(vecProto)).c_str());
    }
}


}  // namespace update
}  // namespace netaos
}  // namespace hozon