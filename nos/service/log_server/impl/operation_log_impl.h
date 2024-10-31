#pragma once

#include <map>

#include "log_server/common/log_server_def.h"

#include "zmq_ipc/manager/zmq_ipc_server.h"
#include "zmq_ipc/proto/log_server.pb.h"
#include "log/src/adapter/hz_operation_log_trace.hpp"

namespace hozon {
namespace netaos {
namespace logserver {

class OperationLogImpl final : public hozon::netaos::zmqipc::ZmqIpcServer
{

public:
    OperationLogImpl();
    virtual ~OperationLogImpl(){};
    int32_t Init();
    int32_t DeInit();
    virtual int32_t Process(const std::string& request, std::string& reply);

private:
    void Logout(const std::string& appid, const std::string& ctxid, LogLevel level, const std::string& message);
    LogLevel convertToLogLevel(uint32_t value);
    std::shared_ptr<hozon::netaos::log::HzOperationLogTrace> GetTraceByCtxID(const std::string& ctxid, const std::string& appid);

private:
   std::mutex mtx_;
   std::map<std::string, std::shared_ptr<hozon::netaos::log::HzOperationLogTrace>> trace_;
};

}  // namespace logserver
}  // namespace netaos
}  // namespace hozon
