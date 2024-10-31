#pragma once

#include "zmq_ipc/manager/zmq_ipc_server.h"

namespace hozon {
namespace netaos {
namespace update {

using namespace hozon::netaos::zmqipc;

class DevmGetVersionMethodServer : public ZmqIpcServer {
public:
    DevmGetVersionMethodServer();
    virtual ~DevmGetVersionMethodServer(){};
    int32_t Init();
    int32_t DeInit();
    virtual int32_t Process(const std::string& request, std::string& reply);
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon