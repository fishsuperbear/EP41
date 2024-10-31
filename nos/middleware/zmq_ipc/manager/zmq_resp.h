
#pragma once

#include "zmq/zmq.hpp"

#include "zmq_ipc/common/zmq_def.h"

namespace hozon {
namespace netaos {
namespace zmqipc {

class ZmqResp {
public:
    ZmqResp();
    ~ZmqResp();

    int32_t Init(const std::string& addr);
    int32_t Start();
    int32_t Stop();

    using ProcessFunc = std::function<int32_t(const std::string&, std::string&)>;
    void RegisterProcess(ProcessFunc func);

private:
    zmq::context_t  context_;
    std::string     endpoint_;
    bool            stopFlag_;

    ProcessFunc     process_;
    std::thread     thread_;
};

}  // namespace zmqipc
}  // namespace netaos
}  // namespace hozon