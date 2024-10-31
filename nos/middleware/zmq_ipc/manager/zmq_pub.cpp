
#include "zmq_ipc/manager/zmq_pub.h"

namespace hozon {
namespace netaos {
namespace zmqipc {

ZmqPub::ZmqPub()
: context_(1)
, endpoint_("")
, stopFlag_(false)
, thread_()
, queue_()
, mutex_()
, condition_()
{
}

int32_t
ZmqPub::Init(const std::string& addr) {
    endpoint_ = addr;
    return endpoint_.size();
}


int32_t
ZmqPub::Start()
{
    stopFlag_ = false;
    thread_ = std::thread([this] {
        zmq::socket_t socket(context_, zmq::socket_type::pub);
        socket.bind(this->endpoint_);

        while (!stopFlag_) {
            std::unique_lock<std::mutex> lock(this->mutex_);
            this->condition_.wait(lock, [&]() {
                return (!this->queue_.empty() || this->stopFlag_);
            }); // 等待新的请求

            if (this->stopFlag_) {
                break;
            }

            std::shared_ptr<std::string> request = this->queue_.front(); // 获取队首请求
            this->queue_.pop_front(); // 出队
            lock.unlock(); // 释放锁

            // 处理请求
            try {
                socket.send(zmq::const_buffer(request->data(), request->size()), zmq::send_flags::none);
            }
            catch (const zmq::error_t& ex) {
                std::cerr << "Server: ZeroMQ Exception: " << ex.what() << std::endl;
            }
        }
        socket.unbind(this->endpoint_);
    });
    return 0;
}

int32_t
ZmqPub::Stop()
{
    stopFlag_ = true;
    if (thread_.joinable()) {
        thread_.join();
    }
    return 0;
}

int32_t
ZmqPub::Publish(const std::shared_ptr<std::string>& data)
{
    // 将新的请求添加到队列中
    std::unique_lock<std::mutex> lock(mutex_);
    queue_.push_back(std::move(data));
    if (queue_.size() == 1) {
        condition_.notify_all();
    }
    return queue_.size();
}


}  // namespace zmqipc
}  // namespace netaos
}  // namespace hozon