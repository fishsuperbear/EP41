#include "update_manager/cmd_line_upgrade/transfer/devm_start_update_method_receiver.h"
#include "update_manager/cmd_line_upgrade/cmd_upgrade_def.h"
#include "update_manager/log/update_manager_logger.h"
#include "update_manager/cmd_line_upgrade/cmd_upgrade_manager.h"
#include "zmq_ipc/proto/devm_tool.pb.h"
namespace hozon {
namespace netaos {
namespace update {

const std::string start_update_service_name = "tcp://*:11133";

DevmStartUpdateMethodServer::DevmStartUpdateMethodServer()
:hozon::netaos::zmqipc::ZmqIpcServer()
{
}

int32_t
DevmStartUpdateMethodServer::Init()
{
    UM_DEBUG << "DevmStartUpdateMethodServer::Init";
    auto res = Start(start_update_service_name);

    return res;
}

int32_t
DevmStartUpdateMethodServer::DeInit()
{
    UM_DEBUG << "DevmStartUpdateMethodServer::DeInit";

    auto res = Stop();
    return res;
}

int32_t 
DevmStartUpdateMethodServer::Process(const std::string& request, std::string& reply)
{
    UPDATE_LOG_D("DevmStartUpdateMethodServer::Process. call CmdUpgradeManager StartUpdateMethod.");
    std::string data(request.begin(), request.end());
    UpgradeUpdateReq req_info{};
    req_info.ParseFromString(data);

    std::shared_ptr<start_update_req_t> data_req = std::make_shared<start_update_req_t>();
    data_req->start_with_precheck = req_info.start_with_precheck();
    data_req->skip_version = req_info.skip_version();
    data_req->package_path = req_info.package_path();
    data_req->ecu_mode = req_info.ecu_mode();

    std::shared_ptr<start_update_resp_t> data_resp = std::make_shared<start_update_resp_t>();
    CmdUpgradeManager::Instance()->StartUpdateMethod(data_req, data_resp);

    UpgradeUpdateResp resp_info{};
    resp_info.set_error_code(data_resp->error_code);
    resp_info.set_error_msg(data_resp->error_msg);

    resp_info.SerializeToString(&reply);
    return 0;
}

}  // namespace update
}  // namespace netaos
}  // namespace hozon