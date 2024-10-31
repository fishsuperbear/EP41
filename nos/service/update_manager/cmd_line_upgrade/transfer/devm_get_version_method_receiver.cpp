#include "update_manager/cmd_line_upgrade/transfer/devm_get_version_method_receiver.h"
#include "update_manager/cmd_line_upgrade/cmd_upgrade_def.h"
#include "update_manager/log/update_manager_logger.h"
#include "update_manager/cmd_line_upgrade/cmd_upgrade_manager.h"
#include "zmq_ipc/proto/devm_tool.pb.h"

namespace hozon {
namespace netaos {
namespace update {

const std::string get_version_service_name = "tcp://*:11134";

DevmGetVersionMethodServer::DevmGetVersionMethodServer()
:hozon::netaos::zmqipc::ZmqIpcServer()
{
}

int32_t
DevmGetVersionMethodServer::Init()
{
    UM_DEBUG << "DevmGetVersionMethodServer::Init";
    auto res = Start(get_version_service_name);

    return res;
}

int32_t
DevmGetVersionMethodServer::DeInit()
{
    UM_DEBUG << "DevmGetVersionMethodServer::DeInit";

    auto res = Stop();
    return res;
}

int32_t 
DevmGetVersionMethodServer::Process(const std::string& request, std::string& reply)
{
    UPDATE_LOG_D("DevmGetVersionMethodServer::Process. call CmdUpgradeManager GetVersionMethod.");
    UpgradeCommonReq req_info{};
    req_info.ParseFromString(request);

    std::shared_ptr<common_req_t> data_req = std::make_shared<common_req_t>();
    data_req->platform = req_info.platform();

    std::shared_ptr<get_version_resp_t> data_resp = std::make_shared<get_version_resp_t>();
    CmdUpgradeManager::Instance()->GetVersionMethod(data_req, data_resp);
    
    UpgradeVersionResp resp_info{};
    resp_info.set_major_version(data_resp->major_version);
    resp_info.set_soc_version(data_resp->soc_version);
    resp_info.set_mcu_version(data_resp->mcu_version);
    resp_info.set_dsv_version(data_resp->dsv_version);
    resp_info.set_swt_version(data_resp->swt_version);
    resp_info.mutable_sensor_version()->insert(data_resp->sensor_version.begin(), data_resp->sensor_version.end());
    resp_info.set_error_code(data_resp->error_code);
    resp_info.set_error_msg(data_resp->error_msg);

    resp_info.SerializeToString(&reply);
    return 0;
}

}  // namespace update
}  // namespace netaos
}  // namespace hozon