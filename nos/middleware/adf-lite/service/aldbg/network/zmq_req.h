#pragma once

#include <iostream>
#include "zmq/zmq.hpp"
#include "zmq/zmq_addon.hpp"

namespace hozon {
namespace netaos {
namespace adf_lite {

class ZmqReq {
public:
    ZmqReq() :
            _context(1),
            _client(_context, zmq::socket_type::req) {

    }

    int32_t Init(const std::string& endpoint) {
        _client.set(zmq::sockopt::linger, 1000);
        _client.connect(endpoint);

        return 0;
    }

    int32_t Request(const std::string& request, std::string& reply) {
        zmq::const_buffer buff(request.data(), request.size());
        _client.send(buff, zmq::send_flags::none);

        // zmq_pollitem_t items[] = {{_client, 0, ZMQ_POLLIN, 0}};
        // zmq::poll(items, 1, std::chrono::milliseconds(timeout_ms));
        // if (items[0].revents & ZMQ_POLLIN) {
            zmq::message_t reply_msg;
            zmq::recv_result_t recv_result;
            recv_result = _client.recv(reply_msg, zmq::recv_flags::none);
            if (!recv_result) {
                return -1;
            }

            reply = reply_msg.to_string();

            return 0;
        // }

        // return -1;
    }

    int32_t Request(const std::string& request, std::string& reply, uint32_t timeout_ms) {
        zmq::const_buffer buff(request.data(), request.size());
        _client.send(buff, zmq::send_flags::none);

        zmq_pollitem_t items[] = {{_client, 0, ZMQ_POLLIN, 0}};
        zmq::poll(items, 1, std::chrono::milliseconds(timeout_ms));
        if (items[0].revents & ZMQ_POLLIN) {
            zmq::message_t reply_msg;
            zmq::recv_result_t recv_result;
            recv_result = _client.recv(reply_msg, zmq::recv_flags::none);
            if (!recv_result) {
                return -1;
            }

            reply = reply_msg.to_string();

            return 0;
        }

        return -1;
    }

    void Close() {
        _client.close();
        _context.close();
    }

private:
    zmq::context_t _context;
    zmq::socket_t _client;
};

}
}
}