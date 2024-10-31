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
    :  skeleton_(nullptr)
{
}

UdsRawDataReqDispatcher::~UdsRawDataReqDispatcher()
{
}

void
UdsRawDataReqDispatcher::Init()
{
    UPDATE_LOG_D("Init");
    std::shared_ptr<uds_raw_data_req_eventPubSubType> pubsubtype_ = std::make_shared<uds_raw_data_req_eventPubSubType>();
    skeleton_ = std::make_shared<Skeleton>(pubsubtype_);
    skeleton_->Init(0, "uds_raw_data_req_eventTopic");
}

void
UdsRawDataReqDispatcher::Deinit()
{
    UPDATE_LOG_D("Deinit");
    if (skeleton_ != nullptr) {
        skeleton_->Deinit();
        skeleton_ = nullptr;
    }
}

void
UdsRawDataReqDispatcher::Send(UdsRawDataReqEvent& sendUdsRawDataReq)
{
    std::shared_ptr<uds_raw_data_req_event> data = std::make_shared<uds_raw_data_req_event>();
    // data->seq_id(sendUdsRawDataReq.seq_id);
    data->sa(sendUdsRawDataReq.sa);
    data->ta(sendUdsRawDataReq.ta);
    data->bus_type(sendUdsRawDataReq.bus_type);
    data->data_vec(sendUdsRawDataReq.data_vec);

    if (skeleton_ == nullptr) {
        UPDATE_LOG_E("skeleton_ is nullptr!.");
        return;
    }

    if (!skeleton_->IsMatched()) {
        UPDATE_LOG_E("skeleton_ is not matched!.");
        return;
    }

    if (skeleton_->Write(data) != 0) {
        UPDATE_LOG_E("UdsRawDataReqDispatcher failed Send --> ReqSa: %X, reqTa: %X, updateType: %d, uds data size: %ld, data: [%s].",
        data->sa(), data->ta(), data->bus_type(), data->data_vec().size(), (UM_UINT8_VEC_TO_HEX_STRING(data->data_vec())).c_str());
    }
    else {
        UPDATE_LOG_D("UdsRawDataReqDispatcher success Send --> ReqSa: %X, reqTa: %X, updateType: %d, uds data size: %ld, data: [%s].",
        data->sa(), data->ta(), data->bus_type(), data->data_vec().size(), (UM_UINT8_VEC_TO_HEX_STRING(data->data_vec())).c_str());
    }
}


}  // namespace update
}  // namespace netaos
}  // namespace hozon