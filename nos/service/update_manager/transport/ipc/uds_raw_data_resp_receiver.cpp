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
{
}

UdsRawDataRespReceiver::~UdsRawDataRespReceiver()
{
}

void
UdsRawDataRespReceiver::Init()
{
    UPDATE_LOG_I("UdsRawDataRespReceiver::Init");
    this->Start("uds_raw_data_resp_eventTopic");
}

void
UdsRawDataRespReceiver::DeInit()
{
    UPDATE_LOG_I("UdsRawDataRespReceiver::DeInit");
    this->Stop();
}

void
UdsRawDataRespReceiver::RegistUdsRawDataReceiveCallback(std::function<void(const std::unique_ptr<uds_raw_data_resp_t>&)> uds_rawdata_receive_callback)
{
    uds_rawdata_receive_callback_ = uds_rawdata_receive_callback;
}

int32_t 
UdsRawDataRespReceiver::Process(const std::vector<uint8_t>& req, std::vector<uint8_t>& resp)
{
    UPDATE_LOG_D("UdsRawDataRespReceiver::Process.");
    std::unique_ptr<uds_raw_data_resp_t> raw_data_resp = std::make_unique<uds_raw_data_resp_t>();

    std::string str(req.begin(), req.end());
    UdsRawDataRespMethod reqProtoData{};
    reqProtoData.ParseFromString(str);

    // req --> reqProtoData  --> uds_raw_data_resp_t
    raw_data_resp->sa = reqProtoData.sa();
    raw_data_resp->ta = reqProtoData.ta();
    raw_data_resp->bus_type = reqProtoData.bus_type();
    raw_data_resp->result = reqProtoData.result();
    raw_data_resp->data_vec.assign(reqProtoData.data_vec().begin(), reqProtoData.data_vec().end());

    if (3 == raw_data_resp->data_vec.size() && raw_data_resp->data_vec[0] == 0x7F && raw_data_resp->data_vec[1] == 0x36 && raw_data_resp->data_vec[2] == 0x78) {
        UPDATE_LOG_D("7F 36 78 no need to handle.");
        return -1;
    }

    if (0 == raw_data_resp->data_vec.size()) {
        UPDATE_LOG_D("no data uds for suppress positive request no need to handle.");
        return -1;
    }

    UPDATE_LOG_D("UdsRawDataRespReceiver Recv <-- ReqTa: %X, reqSa: %X, updateType: %d, result: %d, uds data size: %ld, data: [%s].",
        raw_data_resp->sa, raw_data_resp->ta, raw_data_resp->bus_type, raw_data_resp->result, raw_data_resp->data_vec.size(), (UM_UINT8_VEC_TO_HEX_STRING(raw_data_resp->data_vec)).c_str());

    uds_rawdata_receive_callback_(raw_data_resp);

    resp.clear();
    return 0;
}



}  // namespace update
}  // namespace netaos
}  // namespace hozon
