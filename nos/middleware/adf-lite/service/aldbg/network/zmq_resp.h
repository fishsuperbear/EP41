#pragma once

#include <cstdint>
#include <string>
#include "zmq/zmq.hpp"

namespace hozon {
namespace netaos {
namespace adf_lite {

class ZmqResp {
public:
    ZmqResp() :
            _context(1),
            _server(_context, zmq::socket_type::rep) {

    }

    int32_t Init(const std::string& addr) {
        _server.bind(addr);

        return 0;
    }

    void Close() {
        _server.close();
        _context.close();
    }

    int32_t Recv(std::string& str) {
        zmq::message_t data;
        _server.recv(data);
        str = data.to_string();

        return 0;
    }

    int32_t Send(const std::string& str) {
        zmq::const_buffer data_buff(str.data(), str.size());
        _server.send(data_buff);

        return 0;
    }

private:
    zmq::context_t _context;
    zmq::socket_t _server;
    std::string _topic;
};

}
}
}