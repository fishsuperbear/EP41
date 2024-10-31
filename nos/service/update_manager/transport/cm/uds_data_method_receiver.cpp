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
UdsDataMethodServer::Process(const std::shared_ptr<uds_data_method> req, std::shared_ptr<uds_data_method> resp)
{
    UPDATE_LOG_D("UdsDataMethodServer::Process. call DiagAgent ReceiveUdsData.");
    // req  --> uds_data_req_t
    std::shared_ptr<uds_data_req_t> raw_data_req = std::make_shared<uds_data_req_t>();
    raw_data_req->meta_info = req->meta_info();
    raw_data_req->sid = req->sid();
    raw_data_req->subid = req->subid();
    raw_data_req->data_len = req->data_len();
    raw_data_req->data_vec = req->data_vec();

    std::shared_ptr<uds_data_req_t> raw_data_resp = std::make_shared<uds_data_req_t>();
    DiagAgent::Instance()->ReceiveUdsData(raw_data_req, raw_data_resp);
    // uds_data_req_t --> resp
    resp->meta_info(raw_data_resp->meta_info);
    resp->sid(raw_data_resp->sid);
    resp->subid(raw_data_resp->subid);
    resp->resp_ack(raw_data_resp->resp_ack);
    resp->data_len(raw_data_resp->data_len);
    resp->data_vec(raw_data_resp->data_vec);

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
    std::shared_ptr<uds_data_methodPubSubType> req_data_type = std::make_shared<uds_data_methodPubSubType>();
    std::shared_ptr<uds_data_methodPubSubType> resp_data_type = std::make_shared<uds_data_methodPubSubType>();
    method_server_ = std::make_shared<UdsDataMethodServer>(req_data_type, resp_data_type);
    // method_server_->RegisterProcess(std::bind(&UdsDataMethodServer::Process, method_server_, std::placeholders::_1, std::placeholders::_2));
    method_server_->Start(0, "diag_update");
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