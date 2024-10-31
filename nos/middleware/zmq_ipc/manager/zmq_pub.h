
#pragma once

#include <list>
#include "zmq/zmq.hpp"

#include "zmq_ipc/common/zmq_def.h"

namespace hozon {
namespace netaos {
namespace zmqipc {

class ZmqPub {
public:
    ZmqPub();

    int32_t Init(const std::string& addr);
    int32_t Start();
    int32_t Stop();

    int32_t Publish(const std::shared_ptr<std::string>& data);

private:
    zmq::context_t  context_;
    std::string     endpoint_;
    bool            stopFlag_;

    std::thread     thread_;
    std::list<std::shared_ptr<std::string>>  queue_;
    std::mutex              mutex_;
    std::condition_variable condition_;
};

}  // namespace zmqipc
}  // namespace netaos
}  // namespace hozon