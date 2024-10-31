#include "update_status_method_receiver.h"
#include "update_manager/state_machine/update_state_machine.h"
#include "update_manager/log/update_manager_logger.h"

namespace hozon {
namespace netaos {
namespace update {

using namespace hozon::netaos::diag;


int32_t
UpdateStatusMethodServer::Process(const std::vector<uint8_t>& req, std::vector<uint8_t>& resp)
{
    UPDATE_LOG_D("UpdateStatusMethodServer::Process.");

    resp.clear();
    auto state = UpdateStateMachine::Instance()->GetCurrentState();
    if (state != "NORMAL_IDLE" && state != "OTA_PRE_UPDATE")
    {
        resp.push_back(static_cast<uint8_t>(DiagUpdateStatus::kUpdating));
    } else {
        resp.push_back(static_cast<uint8_t>(DiagUpdateStatus::kUpdated));
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
    method_server_ = std::make_shared<UpdateStatusMethodServer>();
    method_server_->Start("update_status");
}

void
UpdateStatusMethodReceiver::DeInit()
{
    UPDATE_LOG_I("UpdateStatusMethodReceiver::DeInit");
    if (nullptr != method_server_) {
        method_server_->Stop();
        method_server_ = nullptr;
    }
}

}  // namespace update
}  // namespace netaos
}  // namespace hozon