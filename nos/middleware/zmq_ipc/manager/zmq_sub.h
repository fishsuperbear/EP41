
#pragma once

#include "zmq/zmq.hpp"
#include "zmq/zmq_addon.hpp"

#include "zmq_ipc/common/zmq_def.h"

namespace hozon {
namespace netaos {
namespace zmqipc {

class ZmqSub {
public:
    ZmqSub();
    ~ZmqSub();

    int32_t Init(const std::string& endpoint);

    using SubscribeFunc = std::function<int32_t(const std::string&)>;
    int32_t Subscribe(SubscribeFunc func, const std::string& filter = "");
    int32_t Unsubscibe();

private:
    std::string     endpoint_;
    std::string     filter_;
    zmq::context_t  context_;

    bool            stopFlag_;
    SubscribeFunc   subscibe_;
    std::thread     thread_;
};

}  // namespace zmqipc
}  // namespace netaos
}  // namespace hozon
