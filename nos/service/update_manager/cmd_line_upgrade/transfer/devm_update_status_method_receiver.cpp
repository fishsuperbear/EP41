#include "update_manager/cmd_line_upgrade/transfer/devm_update_status_method_receiver.h"
#include "update_manager/cmd_line_upgrade/cmd_upgrade_def.h"
#include "update_manager/log/update_manager_logger.h"
#include "update_manager/cmd_line_upgrade/cmd_upgrade_manager.h"
#include "zmq_ipc/proto/devm_tool.pb.h"
namespace hozon {
namespace netaos {
namespace update {

const std::string status_service_name = "tcp://*:11130";

DevmUpdateStatusMethodServer::DevmUpdateStatusMethodServer()
:hozon::netaos::zmqipc::ZmqIpcServer()
{
}

int32_t
DevmUpdateStatusMethodServer::Init()
{
    UM_DEBUG << "DevmUpdateStatusMethodServer::Init";
    auto res = Start(status_service_name);

    return res;
}

int32_t
DevmUpdateStatusMethodServer::DeInit()
{
    UM_DEBUG << "DevmUpdateStatusMethodServer::DeInit";

    auto res = Stop();
    return res;
}

int32_t 
DevmUpdateStatusMethodServer::Process(const std::string& request, std::string& reply)
{
    UPDATE_LOG_D("DevmUpdateStatusMethodServer::Process. call CmdUpgradeManager UpdateStatusMethod.");
    UpgradeCommonReq req_info{};
    req_info.ParseFromString(request);
    
    std::shared_ptr<common_req_t> data_req = std::make_shared<common_req_t>();
    data_req->platform = req_info.platform();

    std::shared_ptr<update_status_resp_t> data_resp = std::make_shared<update_status_resp_t>();
    CmdUpgradeManager::Instance()->UpdateStatusMethod(data_req, data_resp);

    UpgradeStatusResp resp_info{};
    resp_info.set_update_status(data_resp->update_status);
    resp_info.set_error_code(data_resp->error_code);
    resp_info.set_error_msg(data_resp->error_msg);

    resp_info.SerializeToString(&reply);
    return 0;
}

}  // namespace update
}  // namespace netaos
}  // namespace hozon