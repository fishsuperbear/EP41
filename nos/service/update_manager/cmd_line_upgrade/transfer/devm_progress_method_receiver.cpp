#include "update_manager/cmd_line_upgrade/transfer/devm_progress_method_receiver.h"
#include "update_manager/cmd_line_upgrade/cmd_upgrade_def.h"
#include "update_manager/log/update_manager_logger.h"
#include "update_manager/cmd_line_upgrade/cmd_upgrade_manager.h"
#include "zmq_ipc/proto/devm_tool.pb.h"
namespace hozon {
namespace netaos {
namespace update {

const std::string progress_service_name = "tcp://*:11132";

DevmProgressMethodServer::DevmProgressMethodServer()
:hozon::netaos::zmqipc::ZmqIpcServer()
{
}

int32_t
DevmProgressMethodServer::Init()
{
    UM_DEBUG << "DevmProgressMethodServer::Init";
    auto res = Start(progress_service_name);

    return res;
}

int32_t
DevmProgressMethodServer::DeInit()
{
    UM_DEBUG << "DevmProgressMethodServer::DeInit";

    auto res = Stop();
    return res;
}

int32_t 
DevmProgressMethodServer::Process(const std::string& request, std::string& reply)
{
    UPDATE_LOG_D("DevmProgressMethodServer::Process. call CmdUpgradeManager ProgressMethod.");
    UpgradeCommonReq req_info{};
    req_info.ParseFromString(request);

    std::shared_ptr<common_req_t> data_req = std::make_shared<common_req_t>();
    data_req->platform = req_info.platform();

    std::shared_ptr<progress_resp_t> data_resp = std::make_shared<progress_resp_t>();
    CmdUpgradeManager::Instance()->ProgressMethod(data_req, data_resp);

    UpgradeProgressResp resp_info{};
    resp_info.set_progress(data_resp->progress);
    resp_info.set_error_code(data_resp->error_code);
    resp_info.set_error_msg(data_resp->error_msg);

    resp_info.SerializeToString(&reply);
    return 0;
}

}  // namespace update
}  // namespace netaos
}  // namespace hozon