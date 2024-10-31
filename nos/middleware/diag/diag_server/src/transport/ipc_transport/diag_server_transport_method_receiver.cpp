/*
* Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
* Description: tpl method receiver
*/

#include "diag_server_transport_method_receiver.h"
#include "diag/diag_server/include/common/diag_server_logger.h"

namespace hozon {
namespace netaos {
namespace diag {
namespace cm_transport {

int32_t
DiagServerTransportMethodServer::Process(const std::shared_ptr<uds_data_method> req, std::shared_ptr<uds_data_method> resp)
{
    DG_DEBUG << "DiagServerTransportMethodServer::Process. ";
    return 0;
}

DiagServerTransportMethodServer::~DiagServerTransportMethodServer()
{}

DiagServerTransportMethodReceiver::DiagServerTransportMethodReceiver()
: method_server_(nullptr)
{
}

DiagServerTransportMethodReceiver::~DiagServerTransportMethodReceiver()
{
}

void
DiagServerTransportMethodReceiver::Init()
{
    DG_INFO << "DiagServerTransportMethodReceiver::Init";
    std::shared_ptr<uds_data_methodPubSubType> req_data_type = std::make_shared<uds_data_methodPubSubType>();
    std::shared_ptr<uds_data_methodPubSubType> resp_data_type = std::make_shared<uds_data_methodPubSubType>();
    method_server_ = std::make_shared<DiagServerTransportMethodServer>(req_data_type, resp_data_type);
    // method_server_->RegisterProcess(std::bind(&DiagServerTransportMethodServer::Process, method_server_, std::placeholders::_1, std::placeholders::_2));
    method_server_->Start(0, "diag_tpl");
}

void
DiagServerTransportMethodReceiver::DeInit()
{
    DG_INFO << "DiagServerTransportMethodReceiver::DeInit";
    if (nullptr != method_server_) {
        method_server_->Stop();
        method_server_ = nullptr;
    }
}

}  // namespace cm_transport
}  // namespace diag
}  // namespace netaos
}  // namespace hozon