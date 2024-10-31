
#pragma once

#include "zmq/zmq.hpp"
#include "zmq/zmq_addon.hpp"

#include "zmq_ipc/common/zmq_def.h"

namespace hozon {
namespace netaos {
namespace zmqipc {

class ZmqReq {
public:
    ZmqReq();
    ~ZmqReq();

    int32_t Init(const std::string& endpoint);
    int32_t Deinit();

    int32_t Request(const std::string& request, std::string& reply, uint32_t timeout_ms = 5000);
    int32_t RequestAndForget(const std::string& request);

private:
    std::string endpoint_;
    zmq::context_t context_;
};

}  // namespace zmqipc
}  // namespace netaos
}  // namespace hozon
