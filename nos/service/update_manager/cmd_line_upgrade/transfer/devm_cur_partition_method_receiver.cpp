#include "update_manager/cmd_line_upgrade/transfer/devm_cur_partition_method_receiver.h"
#include "update_manager/cmd_line_upgrade/cmd_upgrade_def.h"
#include "update_manager/log/update_manager_logger.h"
#include "update_manager/cmd_line_upgrade/cmd_upgrade_manager.h"
#include "zmq_ipc/proto/devm_tool.pb.h"
namespace hozon {
namespace netaos {
namespace update {

const std::string cur_partition_service_name = "tcp://*:11136";

DevmCurPartitionMethodServer::DevmCurPartitionMethodServer()
:hozon::netaos::zmqipc::ZmqIpcServer()
{
}

int32_t
DevmCurPartitionMethodServer::Init()
{
    UM_DEBUG << "DevmCurPartitionMethodServer::Init";
    auto res = Start(cur_partition_service_name);

    return res;
}

int32_t
DevmCurPartitionMethodServer::DeInit()
{
    UM_DEBUG << "DevmCurPartitionMethodServer::DeInit";

    auto res = Stop();
    return res;
}

int32_t 
DevmCurPartitionMethodServer::Process(const std::string& request, std::string& reply)
{
    UPDATE_LOG_D("DevmCurPartitionMethodServer::Process. call CmdUpgradeManager PartitionMethod.");
    UpgradeCommonReq req_info{};
    req_info.ParseFromString(request);

    std::shared_ptr<common_req_t> data_req = std::make_shared<common_req_t>();
    data_req->platform = req_info.platform();

    std::shared_ptr<cur_pratition_resp_t> data_resp = std::make_shared<cur_pratition_resp_t>();

    CmdUpgradeManager::Instance()->PartitionMethod(data_req, data_resp);

    UpgradeCurPartitionResp resp_info{};
    resp_info.set_cur_partition(data_resp->cur_partition);
    resp_info.set_error_code(data_resp->error_code);
    resp_info.set_error_msg(data_resp->error_msg);

    resp_info.SerializeToString(&reply);
    return 0;
}

}  // namespace update
}  // namespace netaos
}  // namespace hozon