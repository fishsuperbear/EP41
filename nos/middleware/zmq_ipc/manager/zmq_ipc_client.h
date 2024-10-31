
#pragma once

#include "zmq_ipc/manager/zmq_req.h"

namespace hozon {
namespace netaos {
namespace zmqipc {


class ZmqIpcClient {
public:
    ZmqIpcClient()
    :req_(std::make_unique<ZmqReq>())
    {
    }

    virtual ~ZmqIpcClient(){};

    int32_t Init(const std::string& endpoint) {
        return req_->Init(endpoint);
    }

    int32_t Deinit() {
        return req_->Deinit();
    }

    int32_t Request(const std::string& request, std::string& reply, uint32_t timeout_ms)
    {
        return req_->Request(request, reply, timeout_ms);
    }

    int32_t RequestAndForget(const std::string& request)
    {
        return req_->RequestAndForget(request);
    }

private:
    ZmqIpcClient(const ZmqIpcClient&);
    ZmqIpcClient& operator=(const ZmqIpcClient&);

private:
    std::unique_ptr<ZmqReq> req_;
};

}  // namespace zmqipc
}  // namespace netaos
}  // namespace hozon
