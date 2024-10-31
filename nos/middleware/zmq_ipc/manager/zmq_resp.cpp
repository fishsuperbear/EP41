
#include "zmq_ipc/manager/zmq_resp.h"

namespace hozon {
namespace netaos {
namespace zmqipc {

ZmqResp::ZmqResp()
: context_(1)
, endpoint_("")
, stopFlag_(false)
, process_(nullptr)
, thread_()
{
}

ZmqResp::~ZmqResp()
{
}

int32_t
ZmqResp::Init(const std::string& addr) {
    endpoint_ = addr;
    return endpoint_.size();
}

int32_t
ZmqResp::Start()
{
    stopFlag_ = false;
    thread_ = std::thread([this] {
        zmq::socket_t socket(context_, zmq::socket_type::rep);
        socket.bind(this->endpoint_);
        while (!stopFlag_) {
            std::string req, reply;
            zmq::message_t data;
            try {
                socket.recv(data, zmq::recv_flags::none);
                req = data.to_string();
                if (nullptr != this->process_) {
                    this->process_(req, reply);
                }
                socket.send(zmq::const_buffer(reply.data(), reply.size()), zmq::send_flags::none);
            }
            catch (const zmq::error_t& ex) {
                std::cerr << "Server: ZeroMQ Exception: " << ex.what() << std::endl;
            }
        }
    });
    return 0;
}

int32_t
ZmqResp::Stop()
{
    stopFlag_ = true;
    try {
        context_.shutdown();
    }
    catch (const zmq::error_t& ex) {
        std::cerr << "Client: ZeroMQ Exception: " << ex.what() << std::endl;
    }
    if (thread_.joinable()) {
        thread_.join();
    }
    process_ = nullptr;
    return 0;
}

void
ZmqResp::RegisterProcess(ProcessFunc func)
{
    process_ = std::move(func);
}


}  // namespace zmqipc
}  // namespace netaos
}  // namespace hozon