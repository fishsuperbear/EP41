#include "update_manager/cmd_line_upgrade/transfer/devm_switch_slot_method_receiver.h"
#include "update_manager/cmd_line_upgrade/cmd_upgrade_def.h"
#include "update_manager/log/update_manager_logger.h"
#include "update_manager/cmd_line_upgrade/cmd_upgrade_manager.h"
#include "zmq_ipc/proto/devm_tool.pb.h"
namespace hozon {
namespace netaos {
namespace update {

const std::string switch_slot_service_name = "tcp://*:11137";

DevmSwitchSlotMethodServer::DevmSwitchSlotMethodServer()
:hozon::netaos::zmqipc::ZmqIpcServer()
{
}

int32_t
DevmSwitchSlotMethodServer::Init()
{
    UM_DEBUG << "DevmSwitchSlotMethodServer::Init";
    auto res = Start(switch_slot_service_name);

    return res;
}

int32_t
DevmSwitchSlotMethodServer::DeInit()
{
    UM_DEBUG << "DevmSwitchSlotMethodServer::DeInit";

    auto res = Stop();
    return res;
}

int32_t 
DevmSwitchSlotMethodServer::Process(const std::string& request, std::string& reply)
{
    UPDATE_LOG_D("DevmSwitchSlotMethodServer::Process. call CmdUpgradeManager SwitchSlotMethod.");
    UpgradeCommonReq req_info{};
    req_info.ParseFromString(request);

    std::shared_ptr<common_req_t> data_req = std::make_shared<common_req_t>();
    data_req->platform = req_info.platform();

    std::shared_ptr<switch_slot_resp_t> data_resp = std::make_shared<switch_slot_resp_t>();

    CmdUpgradeManager::Instance()->SwitchSlotMethod(data_req, data_resp);

    UpgradeSwitchSlotResp resp_info{};
    resp_info.set_error_code(data_resp->error_code);
    resp_info.set_error_msg(data_resp->error_msg);

    resp_info.SerializeToString(&reply);
    return 0;
}

}  // namespace update
}  // namespace netaos
}  // namespace hozon