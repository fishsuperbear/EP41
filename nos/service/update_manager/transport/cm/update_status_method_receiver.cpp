#include "update_status_method_receiver.h"
#include "update_manager/state_machine/update_state_machine.h"
#include "update_manager/log/update_manager_logger.h"

namespace hozon {
namespace netaos {
namespace update {

int32_t
UpdateStatusMethodServer::Process(const std::shared_ptr<update_status_method> req, std::shared_ptr<update_status_method> resp)
{
    UPDATE_LOG_D("UpdateStatusMethodServer::Process.");
    // TODO check current status, now return default
    auto state = UpdateStateMachine::Instance()->GetCurrentState();
    if (state != "NORMAL_IDLE" && state != "OTA_PRE_UPDATE")
    {
        resp->update_status(static_cast<uint8_t>(DiagUpdateStatus::kUpdating));
    } else {
        resp->update_status(static_cast<uint8_t>(DiagUpdateStatus::kUpdated));
    }
    return 0;
}

UpdateStatusMethodReceiver::UpdateStatusMethodReceiver()
: method_server_(nullptr)
{
}

UpdateStatusMethodReceiver::~UpdateStatusMethodReceiver()
{
}

void
UpdateStatusMethodReceiver::Init()
{
    UPDATE_LOG_I("UpdateStatusMethodReceiver::Init");
    std::shared_ptr<update_status_methodPubSubType> req_data_type = std::make_shared<update_status_methodPubSubType>();
    std::shared_ptr<update_status_methodPubSubType> resp_data_type = std::make_shared<update_status_methodPubSubType>();
    method_server_ = std::make_shared<UpdateStatusMethodServer>(req_data_type, resp_data_type);
    // method_server_->RegisterProcess(std::bind(&UpdateStatusMethodServer::Process, method_server_, std::placeholders::_1, std::placeholders::_2));
    method_server_->Start(0, "update_status");
}

void
UpdateStatusMethodReceiver::DeInit()
{
    UPDATE_LOG_I("UpdateStatusMethodServer::DeInit");
    if (nullptr != method_server_) {
        method_server_->Stop();
        method_server_ = nullptr;
    }
}

}  // namespace update
}  // namespace netaos
}  // namespace hozon