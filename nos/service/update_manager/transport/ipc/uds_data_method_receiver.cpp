/*
* Copyright (c) Hozon Auto Co., Ltd. 2023-2023. All rights reserved.
* Description: update_manager method receiver
*/

#include "uds_data_method_receiver.h"
#include "update_manager/log/update_manager_logger.h"
#include "update_manager/agent/diag_agent.h"

namespace hozon {
namespace netaos {
namespace update {

int32_t
UdsDataMethodServer::Process(const std::vector<uint8_t>& req, std::vector<uint8_t>& resp)
{
    UPDATE_LOG_D("UdsDataMethodServer::Process. call DiagAgent ReceiveUdsData.");
    std::shared_ptr<uds_data_req_t> raw_data_req = std::make_shared<uds_data_req_t>();

    std::string str(req.begin(), req.end());
    UdsDataMethod reqProtoData{};
    reqProtoData.ParseFromString(str);

    // req --> reqProtoData  --> uds_data_req_t
    for (const auto& entry : reqProtoData.meta_info()) {
        raw_data_req->meta_info[entry.first] = entry.second;
    }
    raw_data_req->sid = reqProtoData.sid();
    raw_data_req->subid = reqProtoData.subid();
    raw_data_req->data_len = reqProtoData.data_len();
    raw_data_req->data_vec.assign(reqProtoData.data_vec().begin(), reqProtoData.data_vec().end());

    std::shared_ptr<uds_data_req_t> raw_data_resp = std::make_shared<uds_data_req_t>();
    DiagAgent::Instance()->ReceiveUdsData(raw_data_req, raw_data_resp);


    // uds_data_req_t --> UdsDataMethod --> resp
    UdsDataMethod respProtoData;
    for (const auto& entry : raw_data_resp->meta_info) {
        (*respProtoData.mutable_meta_info())[entry.first] = entry.second;
    }
    // *respProtoData.mutable_meta_info() = raw_data_resp->meta_info;
    respProtoData.set_sid(raw_data_resp->sid);
    respProtoData.set_subid(raw_data_resp->subid);
    respProtoData.set_resp_ack(raw_data_resp->resp_ack); 
    respProtoData.set_data_len(raw_data_resp->data_len); 
    respProtoData.set_data_vec(raw_data_resp->data_vec.data(), raw_data_resp->data_vec.size());

    std::string serializedData = respProtoData.SerializeAsString();
    resp.assign(serializedData.begin(), serializedData.end());

    return 0;
}

UdsDataMethodReceiver::UdsDataMethodReceiver()
: method_server_(nullptr)
{
}

UdsDataMethodReceiver::~UdsDataMethodReceiver()
{
}

void
UdsDataMethodReceiver::Init()
{
    UPDATE_LOG_I("UdsDataMethodReceiver::Init");
    method_server_ = std::make_shared<UdsDataMethodServer>();
    method_server_->Start("diag_update");
}

void
UdsDataMethodReceiver::DeInit()
{
    UPDATE_LOG_I("UdsDataMethodReceiver::DeInit");
    if (nullptr != method_server_) {
        method_server_->Stop();
        method_server_ = nullptr;
    }
}

}  // namespace update
}  // namespace netaos
}  // namespace hozon