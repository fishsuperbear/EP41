#pragma once

#include "zmq_ipc/manager/zmq_ipc_server.h"

namespace hozon {
namespace netaos {
namespace update {

using namespace hozon::netaos::zmqipc;

class DevmStartFinishMethodServer : public ZmqIpcServer {
public:
    DevmStartFinishMethodServer();
    virtual ~DevmStartFinishMethodServer(){};
    int32_t Init();
    int32_t DeInit();
    virtual int32_t Process(const std::string& request, std::string& reply);
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon