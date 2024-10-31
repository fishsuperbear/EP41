
#pragma once

#include <cstdint>
#include "zmq_ipc/manager/zmq_resp.h"

namespace hozon {
namespace netaos {
namespace zmqipc {

class ZmqIpcServer {
public:
    ZmqIpcServer()
        : resp_(std::make_unique<ZmqResp>()) {}

    virtual ~ZmqIpcServer(){};

    int32_t Start(const std::string& addr) {
        resp_->Init(addr);
        resp_->RegisterProcess([this] (const std::string& request, std::string& reply) {
            return InternalProcess(request, reply);
        });
        return resp_->Start();
    }

    int32_t Stop() {
        return resp_->Stop();
    }

    virtual int32_t Process(const std::string& request, std::string& reply) = 0;

private:
    ZmqIpcServer(const ZmqIpcServer&);
    ZmqIpcServer& operator=(const ZmqIpcServer&);
    int32_t InternalProcess(const std::string& request, std::string& reply) {
        return Process(request, reply);
    }

private:
    std::unique_ptr<ZmqResp> resp_;
};

}  // namespace zmqipc
}  // namespace netaos
}  // namespace hozon