#pragma once

#include <cstdint>
#include <string>
#include "zmq/zmq.hpp"

namespace hozon {
namespace netaos {
namespace adf_lite {

class ZmqPub {
public:
    ZmqPub() :
            _context(1),
            _publisher(_context, ZMQ_PUB) {

    }

    int32_t Init(const std::string& addr) {
        _publisher.bind(addr);

        return 0;
    }

    int32_t Pub(const std::string& topic, const std::string& data) {
        zmq::const_buffer topic_buff(topic.data(), topic.size());
        zmq::const_buffer data_buff(data.data(), data.size());

        if (!_publisher.send(topic_buff, zmq::send_flags::sndmore)) {
            return -1;
        }
        
        if (!_publisher.send(data_buff, zmq::send_flags::none)) {
            return -2;
        }

        return 0;
    }

private:
    zmq::context_t _context;
    zmq::socket_t _publisher;
};

}
}
}