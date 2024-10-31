#include "update_manager/cmd_line_upgrade/transfer/devm_precheck_method_receiver.h"
#include "update_manager/cmd_line_upgrade/cmd_upgrade_def.h"
#include "update_manager/log/update_manager_logger.h"
#include "update_manager/cmd_line_upgrade/cmd_upgrade_manager.h"
#include "zmq_ipc/proto/devm_tool.pb.h"

namespace hozon {
namespace netaos {
namespace update {

const std::string pre_check_service_name = "tcp://*:11131";

DevmPreCheckMethodServer::DevmPreCheckMethodServer()
:hozon::netaos::zmqipc::ZmqIpcServer()
{
}

int32_t
DevmPreCheckMethodServer::Init()
{
    UM_DEBUG << "DevmPreCheckMethodServer::Init";
    auto res = Start(pre_check_service_name);

    return res;
}

int32_t
DevmPreCheckMethodServer::DeInit()
{
    UM_DEBUG << "DevmPreCheckMethodServer::DeInit";

    auto res = Stop();
    return res;
}

int32_t 
DevmPreCheckMethodServer::Process(const std::string& request, std::string& reply)
{
    UPDATE_LOG_D("DevmPreCheckMethodServer::Process. call CmdUpgradeManager PreCheckMethod.");
    UpgradeCommonReq req_info{};
    req_info.ParseFromString(request);

    std::shared_ptr<common_req_t> data_req = std::make_shared<common_req_t>();
    data_req->platform = req_info.platform();
    std::shared_ptr<precheck_resp_t> data_resp = std::make_shared<precheck_resp_t>();
    CmdUpgradeManager::Instance()->PreCheckMethod(data_req, data_resp);

    UpgradePrecheckResp resp_info{};
    resp_info.set_space(data_resp->space);
    resp_info.set_speed(data_resp->speed);
    resp_info.set_gear(data_resp->gear);
    resp_info.set_error_code(data_resp->error_code);
    resp_info.set_error_msg(data_resp->error_msg);

    resp_info.SerializeToString(&reply);
    return 0;
}

}  // namespace update
}  // namespace netaos
}  // namespace hozon