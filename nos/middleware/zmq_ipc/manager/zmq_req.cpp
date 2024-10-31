
#include "zmq_ipc/manager/zmq_req.h"

namespace hozon {
namespace netaos {
namespace zmqipc {

ZmqReq::ZmqReq()
: endpoint_("")
, context_(1)
{
}

ZmqReq::~ZmqReq()
{
}

int32_t
ZmqReq::Init(const std::string& endpoint)
{
    endpoint_ = endpoint;
    return endpoint_.size();
}

int32_t ZmqReq::Deinit()
{
    try {
        context_.shutdown();
    }
    catch (const zmq::error_t& ex) {
        std::cerr << "Client: ZeroMQ Exception: " << ex.what() << std::endl;
    }
    return 0;
}

int32_t
ZmqReq::Request(const std::string& request, std::string& reply, uint32_t timeout_ms)
{
    int32_t ret = -1;
    zmq::recv_result_t recv_result;
    zmq::const_buffer buff(request.data(), request.size());
    zmq::socket_t socket(context_, zmq::socket_type::req);
    uint32_t poll_time = timeout_ms;

    try {
        socket.connect(endpoint_);
        socket.send(buff, zmq::send_flags::none);
        zmq_pollitem_t items[] = {{socket, 0, ZMQ_POLLIN, 0}};
        zmq::poll(items, 1, std::chrono::milliseconds(poll_time));
        if (items[0].revents & ZMQ_POLLIN) {
            zmq::message_t reply_msg;
            socket.recv(reply_msg, zmq::recv_flags::none);
            reply = reply_msg.to_string();
            ret = 0;
        }
    }
    catch (const zmq::error_t& ex) {
        std::cerr << "Client: ZeroMQ Exception: " << ex.what() << std::endl;
    }
    return ret;
}

int32_t
ZmqReq::RequestAndForget(const std::string& request)
{
    std::string reply;
    return this->Request(request, reply, 10);
}

}  // namespace zmqipc
}  // namespace netaos
}  // namespace hozon
