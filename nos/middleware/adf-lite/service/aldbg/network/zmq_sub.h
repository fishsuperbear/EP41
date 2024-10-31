#pragma once

#include <cstdint>
#include <string>
#include "zmq/zmq.hpp"

namespace hozon {
namespace netaos {
namespace adf_lite {

class ZmqSub {
public:
    ZmqSub() :
            _context(1),
            _subscriber(_context, ZMQ_SUB) {

    }

    int32_t Init(const std::string& addr, const std::string& topic) {
        _subscriber.connect(addr);
        _subscriber.set(zmq::sockopt::subscribe, topic.data());
        _topic = topic;

        return 0;
    }

    int32_t Recv(zmq::message_t& msg) {
        _subscriber.recv(msg);

        return 0;
    }

    int32_t Recv(std::string& str) {
        zmq::message_t topic;
        
        _subscriber.recv(topic);
        if (topic.to_string() != _topic) {
            return -1;
        }

        zmq::message_t data;
        _subscriber.recv(data);
        str = data.to_string();
        return 0;
    }

    void Close() {
        _subscriber.close();
        _context.close();
    }

private:
    zmq::context_t _context;
    zmq::socket_t _subscriber;
    std::string _topic;
};

}
}
}