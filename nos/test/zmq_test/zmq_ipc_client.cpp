#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>
#include <signal.h>


#include "zmq_ipc/manager/zmq_ipc_client.h"

#include "zmq/zmq.hpp"
#include "log/include/default_logger.h"

bool stopFlag_ =  false;

zmq::context_t context1(1), context2(1), context3(1);
zmq::context_t g_client_contex(1);
zmq::socket_t  g_req_socket(g_client_contex, zmq::socket_type::req);
void SigHandler(int signum)
{
    DEBUG_LOG("sigHandler signum: %d", signum);
    DEBUG_LOG("g_req_socket.close()~");
    g_req_socket.close();
    DEBUG_LOG("g_client_contex.shutdown()~");
    g_client_contex.shutdown();
    DEBUG_LOG("stopFlag_ = true~");
    stopFlag_ = true;
}

void ClientAlsoServerMode()
{
    std::thread rep_thread([&context1, &stopFlag_] {
        std::string zmq_requester = "tcp://localhost:57777";

        zmq::socket_t  socket1(context1, zmq::socket_type::req);
        zmq::socket_t  socket2(context1, zmq::socket_type::req);
        zmq::socket_t  socket3(context1, zmq::socket_type::req);

        DEBUG_LOG("client connect zmq_requester begin");
        socket1.connect(zmq_requester);
        socket2.connect(zmq_requester);
        socket3.connect(zmq_requester);
        DEBUG_LOG("client connect zmq_requester end");

        std::string req, reply;
        zmq::message_t data;
        zmq::send_result_t res;
        while (!stopFlag_) {
            req = "client1: 123";
            res = socket1.send(zmq::const_buffer(req.data(), req.size()), zmq::send_flags::none);
            DEBUG_LOG("CLIENT1 after send before recv");
            socket1.recv(data, zmq::recv_flags::none);
            reply = data.to_string();
            DEBUG_LOG("CLIENT1 Req req: %s, reply: %s, res: %d, msg: %s", req.c_str(), reply.c_str(), zmq_errno(), zmq_strerror(zmq_errno()));

            req = "client2: 456456";
            res = socket2.send(zmq::const_buffer(req.data(), req.size()), zmq::send_flags::none);
            DEBUG_LOG("CLIENT2 after send before recv");
            socket2.recv(data);
            reply = data.to_string();
            DEBUG_LOG("CLIENT2 Req req: %s, reply: %s, res: %d, msg: %s", req.c_str(), reply.c_str(), zmq_errno(), zmq_strerror(zmq_errno()));

            req = "client3: 789789789";
            res = socket3.send(zmq::const_buffer(req.data(), req.size()), zmq::send_flags::none);
            DEBUG_LOG("CLIENT3 after send before recv");
            socket3.recv(data);
            reply = data.to_string();
            DEBUG_LOG("CLIENT3 Req req: %s, reply: %s, res: %d, msg: %s", req.c_str(), reply.c_str(), zmq_errno(), zmq_strerror(zmq_errno()));

            std::this_thread::sleep_for(std::chrono::milliseconds(5000));
        }
        DEBUG_LOG("ClientPullSyncMode");
        socket1.disconnect(zmq_requester);
        socket2.disconnect(zmq_requester);
        socket3.disconnect(zmq_requester);
        DEBUG_LOG("ClientPullSyncMode");
    });
    while (!stopFlag_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}


void ClientSubSyncMode()
{
    bool isConnected = false;
    std::thread rep_thread([&context1, &stopFlag_, &isConnected] {
        std::string zmq_requester = "tcp://localhost:57777";

        zmq::socket_t  socket1(context1, zmq::socket_type::req);
        zmq::socket_t  socket2(context1, zmq::socket_type::req);
        zmq::socket_t  socket3(context1, zmq::socket_type::req);

        DEBUG_LOG("client connect zmq_requester begin");
        socket1.connect(zmq_requester);
        socket2.connect(zmq_requester);
        socket3.connect(zmq_requester);
        DEBUG_LOG("client connect zmq_requester end, !(socket): %d, connected: %d", !(socket1), socket1.connected());

        std::string req, reply;
        zmq::message_t data;
        zmq::send_result_t res;
        while (!stopFlag_ ) {

            if (isConnected) {
                req = "client1: 123";
                res = socket1.send(zmq::const_buffer(req.data(), req.size()), zmq::send_flags::none);
                DEBUG_LOG("CLIENT1 after send before recv");
                socket1.recv(data, zmq::recv_flags::none);
                reply = data.to_string();
                DEBUG_LOG("CLIENT1 Req req: %s, reply: %s, res: %d, msg: %s", req.c_str(), reply.c_str(), zmq_errno(), zmq_strerror(zmq_errno()));

                req = "client2: 456456";
                res = socket2.send(zmq::const_buffer(req.data(), req.size()), zmq::send_flags::none);
                DEBUG_LOG("CLIENT2 after send before recv");
                socket2.recv(data);
                reply = data.to_string();
                DEBUG_LOG("CLIENT2 Req req: %s, reply: %s, res: %d, msg: %s", req.c_str(), reply.c_str(), zmq_errno(), zmq_strerror(zmq_errno()));

                req = "client3: 789789789";
                res = socket3.send(zmq::const_buffer(req.data(), req.size()), zmq::send_flags::none);
                DEBUG_LOG("CLIENT3 after send before recv");
                socket3.recv(data);
                reply = data.to_string();
                DEBUG_LOG("CLIENT3 Req req: %s, reply: %s, res: %d, msg: %s", req.c_str(), reply.c_str(), zmq_errno(), zmq_strerror(zmq_errno()));

            }

            std::this_thread::sleep_for(std::chrono::milliseconds(5000));
        }
        DEBUG_LOG("ClientPullSyncMode");
        socket1.disconnect(zmq_requester);
        socket2.disconnect(zmq_requester);
        socket3.disconnect(zmq_requester);
        DEBUG_LOG("ClientPullSyncMode");
    });

    std::thread sub_thread([&context1, &stopFlag_, &isConnected] {
        std::string zmq_publier = "tcp://localhost:57778";
        DEBUG_LOG("ClientSubSyncMode");
        zmq::socket_t  socket4(context1, zmq::socket_type::sub);
        socket4.connect(zmq_publier);
        DEBUG_LOG("ClientSubSyncMode");
        socket4.set(zmq::sockopt::subscribe, "s");
        DEBUG_LOG("ClientSubSyncMode");
        while (!stopFlag_) {
            std::string reply, req;
            zmq::message_t data;
            zmq::send_result_t res3 = socket4.recv(data, zmq::recv_flags::none);
            req = data.to_string();
            isConnected = true;
            DEBUG_LOG("CLIENT4 Sub req: %s, reply: %s, res: %d, isConnected: %d, msg: %s", req.c_str(), reply.c_str(), zmq_errno(), isConnected, zmq_strerror(zmq_errno()));
        }
        DEBUG_LOG("ClientSubSyncMode");
        socket4.disconnect(zmq_publier);
        DEBUG_LOG("ClientSubSyncMode");
    });


    while (!stopFlag_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}


void ClientPullerSubMode()
{
    std::thread sub_thread([&context1, &stopFlag_] {
        std::string zmq_publier = "tcp://localhost:57778";
        DEBUG_LOG("ClientPublierAsyncMode");
        zmq::socket_t  socket4(context1, zmq::socket_type::sub);
        socket4.connect(zmq_publier);
        DEBUG_LOG("ClientPublierAsyncMode");
        socket4.set(zmq::sockopt::subscribe, "s");
        DEBUG_LOG("ClientPublierAsyncMode");
        while (!stopFlag_) {
            std::string reply, req;
            zmq::message_t data;
            zmq::send_result_t res3 = socket4.recv(data, zmq::recv_flags::none);
            req = data.to_string();
            DEBUG_LOG("CLIENT4 Sub req: %s, reply: %s, res: %d, msg: %s", req.c_str(), reply.c_str(), zmq_errno(), zmq_strerror(zmq_errno()));
        }
        DEBUG_LOG("ClientPublierAsyncMode");
        socket4.disconnect(zmq_publier);
        DEBUG_LOG("ClientPublierAsyncMode");
    });

    std::thread work_threads[5];
    for (int i = 0; i < 5; ++i) {
        work_threads[i] = std::thread([&context1, &stopFlag_] {
            std::string zmq_puller = "tcp://localhost:57779";
            DEBUG_LOG("ClientPullAsyncMode");
            zmq::socket_t  socket5(context1, zmq::socket_type::pull);
            socket5.connect(zmq_puller);
            DEBUG_LOG("ClientPullAsyncMode");
            while (!stopFlag_) {
                std::string reply, req;
                zmq::message_t data;
                zmq::send_result_t res3 = socket5.recv(data, zmq::recv_flags::none);
                req = data.to_string();
                DEBUG_LOG("CLIENT5 Pull recv: %s, reply: %s, res: %d, msg: %s", req.c_str(), reply.c_str(), zmq_errno(), zmq_strerror(zmq_errno()));
            }
            DEBUG_LOG("ClientPullAsyncMode");
            socket5.disconnect(zmq_puller);
            DEBUG_LOG("ClientPullAsyncMode");
        });
    }
    while (!stopFlag_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}


void ClientInterfaceMode()
{


    std::thread rep_thread([&g_req_socket, &stopFlag_] {
        std::string zmq_requester = "tcp://localhost:57777";

        DEBUG_LOG("client connect zmq_requester begin");
        g_req_socket.connect(zmq_requester);

        DEBUG_LOG("client connect zmq_requester end");

        std::string req, reply;
        zmq::message_t data;
        zmq::send_result_t res;
        while (!stopFlag_) {
            req = "client1: 123";

            try {
                res = g_req_socket.send(zmq::const_buffer(req.data(), req.size()), zmq::send_flags::none);
                DEBUG_LOG("CLIENT1 after send before recv");

                zmq_pollitem_t items[] = {{g_req_socket, 0, ZMQ_POLLIN, 0}};
                zmq::poll(items, 1, std::chrono::milliseconds(3000));
                if (items[0].revents & ZMQ_POLLIN) {
                    g_req_socket.recv(data, zmq::recv_flags::none);
                    reply = data.to_string();
                    DEBUG_LOG("CLIENT1 Req req: %s, reply: %s, res: %d, msg: %s", req.c_str(), reply.c_str(), zmq_errno(), zmq_strerror(zmq_errno()));

                }

                // g_req_socket.recv(data, zmq::recv_flags::dontwait);
                // reply = data.to_string();
                // DEBUG_LOG("CLIENT1 Req req: %s, reply: %s, res: %d, msg: %s", req.c_str(), reply.c_str(), zmq_errno(), zmq_strerror(zmq_errno()));
            }
            catch (const zmq::error_t& ex) {
                DEBUG_LOG("Client: ZeroMQ Exception: %s", ex.what());
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5000));
        }
        DEBUG_LOG("ClientPullSyncMode");
        g_req_socket.disconnect(zmq_requester);
        DEBUG_LOG("ClientPullSyncMode");
    });


    while (!stopFlag_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}


int main(int argc, char ** argv) {
    signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);

    // ClientAlsoServerMode();
    // ClientSubSyncMode();
    // ClientPullerSubMode();
    ClientInterfaceMode();

    // std::string compress_log_service_name = "tcp://localhost:55778";
    // std::unique_ptr<hozon::netaos::zmqipc::ZmqIpcClient>   client_ = std::make_unique<hozon::netaos::zmqipc::ZmqIpcClient>();
    // client_->Init(compress_log_service_name);
    // client_->RequestAndForget("Hello");

    // while (!stopFlag_) {
    //     client_->RequestAndForget("Hello");
    //     DEBUG_LOG("client_->RequestAndForget request: Hello!~");
    //     std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    // }


    // DEBUG_LOG("client_->Deinit before!~");
    // client_->Deinit();
    DEBUG_LOG("client_->Deinit end!~");
    // raise(SIGINT);

    return 0;
}
