/*
* Copyright (c) Hozon Auto Co., Ltd. 2023-2023. All rights reserved.
* Description: uds raw data event receiver
*/


#include "uds_raw_data_resp_receiver.h"
#include "update_manager/log/update_manager_logger.h"

namespace hozon {
namespace netaos {
namespace update {


UdsRawDataRespReceiver::UdsRawDataRespReceiver()
: proxy_(nullptr)
, data_(nullptr)
{
}

UdsRawDataRespReceiver::~UdsRawDataRespReceiver()
{
}

void
UdsRawDataRespReceiver::Init()
{
    UPDATE_LOG_I("UdsRawDataRespReceiver::Init");
    std::shared_ptr<uds_raw_data_resp_eventPubSubType> pubsubtype_ = std::make_shared<uds_raw_data_resp_eventPubSubType>();
    proxy_ = std::make_shared<Proxy>(pubsubtype_);
    proxy_->Init(0, "uds_raw_data_resp_eventTopic");
    data_ = std::make_shared<uds_raw_data_resp_event>();

    proxy_->Listen(std::bind(&UdsRawDataRespReceiver::EventCallback, this));
}

void
UdsRawDataRespReceiver::DeInit()
{
    UPDATE_LOG_I("UdsRawDataRespReceiver::DeInit");
    if (nullptr != proxy_) {
        proxy_->Deinit();
        proxy_ = nullptr;
    }
    data_ = nullptr;
}

void
UdsRawDataRespReceiver::RegistUdsRawDataReceiveCallback(std::function<void(const std::unique_ptr<uds_raw_data_resp_t>&)> uds_rawdata_receive_callback)
{
    uds_rawdata_receive_callback_ = uds_rawdata_receive_callback;
}

void
UdsRawDataRespReceiver::RegistReadVersionReceiveCallback(std::function<void(const std::unique_ptr<uds_raw_data_resp_t>&)> read_version_callback)
{
    read_version_callback_ = read_version_callback;
}

void
UdsRawDataRespReceiver::EventCallback()
{
    if (proxy_ != nullptr && proxy_->IsMatched()) {
        proxy_->Take(data_);
        if (3 == data_->data_vec().size() && data_->data_vec()[0] == 0x7F && data_->data_vec()[1] == 0x36 && data_->data_vec()[2] == 0x78) {
            UPDATE_LOG_D("7F 36 78 no need to handle.");
            return;
        }

        if (0 == data_->data_vec().size()) {
            UPDATE_LOG_D("no data uds for suppress positive request no need to handle.");
            return;
        }

        UPDATE_LOG_D("UdsRawDataRespReceiver Recv <-- ReqTa: %X, reqSa: %X, updateType: %d, result: %d, uds data size: %ld, data: [%s].",
            data_->sa(), data_->ta(), data_->bus_type(), data_->result(), data_->data_vec().size(), (UM_UINT8_VEC_TO_HEX_STRING(data_->data_vec())).c_str());

        std::unique_ptr<uds_raw_data_resp_t> uds_data = std::make_unique<uds_raw_data_resp_t>();
        uds_data->bus_type = data_->bus_type();
        uds_data->result = data_->result();
        uds_data->sa = data_->sa();
        uds_data->ta = data_->ta();
        uds_data->data_vec = data_->data_vec();
        uds_rawdata_receive_callback_(uds_data);
        read_version_callback_(uds_data);
    }
}


}  // namespace update
}  // namespace netaos
}  // namespace hozon
