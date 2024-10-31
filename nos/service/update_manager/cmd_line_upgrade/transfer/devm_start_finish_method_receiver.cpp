#include "update_manager/cmd_line_upgrade/transfer/devm_start_finish_method_receiver.h"
#include "update_manager/cmd_line_upgrade/cmd_upgrade_def.h"
#include "update_manager/log/update_manager_logger.h"
#include "update_manager/cmd_line_upgrade/cmd_upgrade_manager.h"
#include "zmq_ipc/proto/devm_tool.pb.h"

namespace hozon {
namespace netaos {
namespace update {

const std::string finish_service_name = "tcp://*:11135";

DevmStartFinishMethodServer::DevmStartFinishMethodServer()
:hozon::netaos::zmqipc::ZmqIpcServer()
{
}

int32_t
DevmStartFinishMethodServer::Init()
{
    UM_DEBUG << "DevmStartFinishMethodServer::Init";
    auto res = Start(finish_service_name);

    return res;
}

int32_t
DevmStartFinishMethodServer::DeInit()
{
    UM_DEBUG << "DevmStartFinishMethodServer::DeInit";

    auto res = Stop();
    return res;
}

int32_t 
DevmStartFinishMethodServer::Process(const std::string& request, std::string& reply)
{
    UPDATE_LOG_D("DevmStartFinishMethodServer::Process. call CmdUpgradeManager StartFinishMethod.");
    UpgradeCommonReq req_info{};
    req_info.ParseFromString(request);

    std::shared_ptr<common_req_t> data_req = std::make_shared<common_req_t>();
    data_req->platform = req_info.platform();

    std::shared_ptr<start_finish_resp_t> data_resp = std::make_shared<start_finish_resp_t>();
    CmdUpgradeManager::Instance()->StartFinishMethod(data_req, data_resp);

    UpgradeFinishResp resp_info{};
    resp_info.set_error_code(data_resp->error_code);
    resp_info.set_error_msg(data_resp->error_msg);

    resp_info.SerializeToString(&reply);
    return 0;
}

}  // namespace update
}  // namespace netaos
}  // namespace hozon