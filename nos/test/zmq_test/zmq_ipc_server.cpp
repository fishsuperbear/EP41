#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>
#include <signal.h>

#include "log/include/default_logger.h"

#include "zmq/zmq.hpp"

#include "zmq_ipc/manager/zmq_ipc_server.h"

bool stopFlag_ =  false;
zmq::context_t g_server_contex(1);
zmq::socket_t  g_rep_socket(g_server_contex, zmq::socket_type::rep);
void SigHandler(int signum)
{
    DEBUG_LOG("sigHandler signum: %d", signum);
    DEBUG_LOG("g_rep_socket.close()~");
    g_rep_socket.close();
    DEBUG_LOG("g_server_contex.shutdown()~");
    g_server_contex.shutdown();
    DEBUG_LOG("stopFlag_ = true~");
    stopFlag_ = true;
}

void ServerAlsoClientMode()
{
    zmq::context_t context1(1);

    DEBUG_LOG("server start recv thread");
    std::thread server_thread([&context1, &stopFlag_] {
        std::string zmq_server = "tcp://*:57777";
        zmq::socket_t  socket3(context1, zmq::socket_type::rep);
        socket3.bind(zmq_server);
        while (!stopFlag_) {
            std::string reply, req;
            zmq::message_t data;
            zmq::send_result_t res3 = socket3.recv(data, zmq::recv_flags::none);
            req = data.to_string();
            socket3.send(data);
            DEBUG_LOG("SERVER3 RECV req: %s, reply: %s, res: %d, msg: %s", req.c_str(), reply.c_str(), zmq_errno(), zmq_strerror(zmq_errno()));
        }

        DEBUG_LOG("server thread quit");
    });

    DEBUG_LOG("server send self messgages");

    std::thread client_thread([&context1, &stopFlag_] {
        std::string zmq_client = "tcp://localhost:57777";
        zmq::socket_t  socket1(context1, zmq::socket_type::req);
        zmq::socket_t  socket2(context1, zmq::socket_type::req);
        socket1.connect(zmq_client);
        socket2.connect(zmq_client);
        while (!stopFlag_) {
            std::string req, reply;
            zmq::message_t data;
            req = "server1: 123";
            auto res = socket1.send(zmq::const_buffer(req.data(), req.size()), zmq::send_flags::none);
            socket1.recv(data);
            reply = data.to_string();
            DEBUG_LOG("SEVER1 send req: %s, reply: %s, res: %d, msg: %s", req.c_str(), reply.c_str(), zmq_errno(), zmq_strerror(zmq_errno()));

            req = "server2: 456456";
            res = socket2.send(zmq::const_buffer(req.data(), req.size()), zmq::send_flags::none);
            socket2.recv(data);
            reply = data.to_string();
            DEBUG_LOG("SERVER2 send req: %s, reply: %s, res: %d, msg: %s", req.c_str(), reply.c_str(), zmq_errno(), zmq_strerror(zmq_errno()));

            std::this_thread::sleep_for(std::chrono::milliseconds(5000));
        }

        DEBUG_LOG("server disconnect and unbind zmq_server begin");
        socket1.disconnect(zmq_client);
        socket2.disconnect(zmq_client);
        DEBUG_LOG("server disconnect and unbind zmq_server end");

    });

    while (!stopFlag_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

void ServerWorkersMode()
{
    std::string zmq_client = "tcp://localhost:57777";
    std::string zmq_server = "tcp://*:57777";
    zmq::context_t context1(1);
    zmq::socket_t socket4(context1, zmq::socket_type::router);
    socket4.bind(zmq_server);
    DEBUG_LOG("server router dealer bind");

    std::string zmq_worker = "inproc://workers";
    zmq::socket_t socket5(context1, zmq::socket_type::dealer);
    socket5.bind(zmq_worker);

    DEBUG_LOG("server start worker threads");
    std::thread work_threads[5];
    for (int i = 0; i < 5; ++i) {
        work_threads[i] = std::thread([&context1] {
            std::string zmq_worker = "inproc://workers";
            zmq::socket_t socket6(context1, zmq::socket_type::rep);
            socket6.connect(zmq_worker);
            DEBUG_LOG("server start worker threads");
            while (!stopFlag_) {
                std::string reply, req;
                zmq::message_t data;
                zmq::send_result_t res3 = socket6.recv(data, zmq::recv_flags::none);
                req = data.to_string();
                socket6.send(data);
                DEBUG_LOG("SERVER6 RECV req: %s, reply: %s, res: %d, msg: %s", req.c_str(), reply.c_str(), zmq_errno(), zmq_strerror(zmq_errno()));
            };
            socket6.unbind(zmq_worker);
        });
    }

    DEBUG_LOG("server start worker threads");

    // 启动队列装置
    zmq::proxy(socket4, socket5);

    DEBUG_LOG("server disconnect and unbind zmq_server begin");
    socket4.unbind(zmq_server);
    socket5.unbind(zmq_server);
    DEBUG_LOG("server disconnect and unbind zmq_server end");
}

void ServerPubSyncMode()
{
    DEBUG_LOG("ServerPubSyncMode");

    zmq::context_t context1(1), context2(1);

    bool isConnected = false;

    std::thread pub_thread([&context1, &stopFlag_ ] {
        DEBUG_LOG("publisher bind");
        std::string zmq_publier = "tcp://*:57778";
        zmq::socket_t socket7(context1, zmq::socket_type::pub);
        socket7.bind(zmq_publier);
        DEBUG_LOG("publisher bind");
        socket7.bind("ipc://weather.ipc");
        DEBUG_LOG("publisher bind");
        std::string req, reply;
        zmq::message_t data;
        while (!stopFlag_) {
            req = "server7: publisher 000000";
            auto res = socket7.send(zmq::const_buffer(req.data(), req.size()), zmq::send_flags::none);
            DEBUG_LOG("SERVER7 Pub req: %s, reply: %s, res: %d, msg: %s", req.c_str(), reply.c_str(), zmq_errno(), zmq_strerror(zmq_errno()));
            std::this_thread::sleep_for(std::chrono::milliseconds(5000));
        }
        DEBUG_LOG("ServerPubSyncMode");
        socket7.unbind(zmq_publier);
        DEBUG_LOG("ServerPubSyncMode");
    });

    std::thread req_thread([&context1, &stopFlag_] {
        int subcnt = 0;
        std::string zmq_server = "tcp://*:57777";
        std::string reply, req;
        zmq::message_t data;
        zmq::socket_t socket6(context1, zmq::socket_type::rep);
        socket6.bind(zmq_server);
        while (!stopFlag_) {
            zmq::send_result_t res3 = socket6.recv(data, zmq::recv_flags::none);
            req = data.to_string();
            socket6.send(data);
            DEBUG_LOG("SERVER6 Rep subcnt: %d, req: %s, reply: %s, res: %d, msg: %s", subcnt, req.c_str(), reply.c_str(), zmq_errno(), zmq_strerror(zmq_errno()));
        }
    });


    DEBUG_LOG("ServerPublierSyncMode");
    while (!stopFlag_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

void ServerPullerPublisherMode()
{
    DEBUG_LOG("PullerPublisherMode");

    zmq::context_t context1(1), context2(1);

    std::thread push_thread([&context1, &stopFlag_ ] {
        DEBUG_LOG("puller bind");
        std::string zmq_pusher = "tcp://*:57779";
        zmq::socket_t socket8(context1, zmq::socket_type::push);
        socket8.bind(zmq_pusher);

        std::string req, reply;
        zmq::message_t data;
        while (!stopFlag_) {
            req = "server8: pusher 000000";
            auto res = socket8.send(zmq::const_buffer(req.data(), req.size()), zmq::send_flags::none);
            DEBUG_LOG("SERVER8 Push req: %s, reply: %s, res: %d, msg: %s", req.c_str(), reply.c_str(), zmq_errno(), zmq_strerror(zmq_errno()));
        }
        DEBUG_LOG("ServerPushAsyncMode");
        socket8.unbind(zmq_pusher);
        DEBUG_LOG("ServerPushAsyncMode");
    });


    std::thread pub_thread([&context1, &stopFlag_ ] {
        DEBUG_LOG("publisher bind");
        std::string zmq_publier = "tcp://*:57778";
        zmq::socket_t socket7(context1, zmq::socket_type::pub);
        socket7.bind(zmq_publier);
        DEBUG_LOG("publisher bind");
        socket7.bind("ipc://weather.ipc");
        DEBUG_LOG("publisher bind");
        std::string req, reply;
        zmq::message_t data;
        while (!stopFlag_) {
            req = "server7: publisher 000000";
            auto res = socket7.send(zmq::const_buffer(req.data(), req.size()), zmq::send_flags::none);
            DEBUG_LOG("SERVER7 Pub req: %s, reply: %s, res: %d, msg: %s", req.c_str(), reply.c_str(), zmq_errno(), zmq_strerror(zmq_errno()));
        }
        DEBUG_LOG("ServerPublierSyncMode");
        socket7.unbind(zmq_publier);
        DEBUG_LOG("ServerPublierSyncMode");
    });
    while (!stopFlag_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

void ServerInterfaceMode()
{
    std::thread push_thread([&g_rep_socket, &stopFlag_ ] {
        DEBUG_LOG("rep bind");
        std::string zmq_server = "tcp://*:57777";
        g_rep_socket.bind(zmq_server);

        std::string req, reply;
        zmq::message_t data;
        while (!stopFlag_) {
            zmq::send_result_t res3 = g_rep_socket.recv(data, zmq::recv_flags::none);
            DEBUG_LOG("SERVER6 after recv before send");
            req = data.to_string();
            g_rep_socket.send(data);
            DEBUG_LOG("SERVER6 Rep  req: %s, reply: %s, res: %d, msg: %s", req.c_str(), reply.c_str(), zmq_errno(), zmq_strerror(zmq_errno()));
        }
        DEBUG_LOG("ServerInterfaceMode");
        g_rep_socket.unbind(zmq_server);
        DEBUG_LOG("ServerInterfaceMode");
    });


    while (!stopFlag_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

}


class CompressLogImpl final : public hozon::netaos::zmqipc::ZmqIpcServer
{

public:
    CompressLogImpl()
    : hozon::netaos::zmqipc::ZmqIpcServer()
    {
    }

    virtual ~CompressLogImpl(){};

    virtual int32_t Process(const std::string& request, std::string& reply)
    {
        reply = "world!";
        DEBUG_LOG("Server recv request: %s, reply: %s", request.c_str(), reply.c_str());
        return reply.size();
    };

private:

};


int main(int argc, char ** argv) {
    signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);

    // ServerAlsoClientMode();
    // ServerWorkersMode();
    // ServerPubSyncMode();
    // ServerPullerPublisherMode();
    ServerInterfaceMode();

    // std::string compress_log_service_name = "tcp://*:55778";
    // std::unique_ptr<CompressLogImpl> server_ = std::make_unique<CompressLogImpl>();
    // server_->Start(compress_log_service_name);

    while (!stopFlag_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // DEBUG_LOG("client_->Stop before!~");
    // server_->Stop();
    // DEBUG_LOG("client_->Stop after!~");

    DEBUG_LOG("Deinit end~");
    return 0;
}
