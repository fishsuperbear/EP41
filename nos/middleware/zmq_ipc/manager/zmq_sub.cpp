
#include "zmq_ipc/manager/zmq_sub.h"

namespace hozon {
namespace netaos {
namespace zmqipc {

ZmqSub::ZmqSub()
: endpoint_("")
, filter_("")
, context_(1)
, stopFlag_(false)
, subscibe_(nullptr)
, thread_()
{
}

ZmqSub::~ZmqSub()
{
}

int32_t
ZmqSub::Init(const std::string& endpoint)
{
    endpoint_ = endpoint;
    return endpoint_.size();
}

int32_t
ZmqSub::Subscribe(SubscribeFunc func, const std::string& filter)
{
    stopFlag_ = false;
    subscibe_ = func;
    filter_ = filter;
    thread_ = std::thread([this] {
        zmq::socket_t  socket(this->context_, zmq::socket_type::sub);
        socket.connect(this->endpoint_);
        socket.set(zmq::sockopt::subscribe, filter_);
        while (!stopFlag_) {
            zmq::message_t data;
            try {
                socket.recv(data, zmq::recv_flags::none);
                if (data.size() > 0 && nullptr != this->subscibe_) {
                    this->subscibe_(data.to_string());
                }
            }
            catch (const zmq::error_t& ex) {
                std::cerr << "Server: ZeroMQ Exception: " << ex.what() << std::endl;
            }
        }
        socket.disconnect(this->endpoint_);
    });

    return 0;
}

int32_t ZmqSub::Unsubscibe()
{
    stopFlag_ = true;
    if (thread_.joinable()) {
        thread_.join();
    }
    subscibe_ = nullptr;
    return 0;
}

}  // namespace zmqipc
}  // namespace netaos
}  // namespace hozon
