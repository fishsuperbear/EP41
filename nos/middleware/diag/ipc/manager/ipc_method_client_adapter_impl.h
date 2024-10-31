#pragma once

#include "diag/ipc/common/ipc_def.h"
namespace hozon {
namespace netaos {
namespace diag {

class IPCMethodClientAdapterImpl {
public:
    IPCMethodClientAdapterImpl();
            
    ~IPCMethodClientAdapterImpl();
    
    int32_t Init(const std::string& service_name);
    int32_t Deinit();
    int32_t Request(const std::vector<uint8_t>& req, std::vector<uint8_t>& resp, const int64_t& timeout_ms);
    int32_t RequestAndForget(const std::vector<uint8_t>& req);
    int32_t IsMatched();
private:
    void CheckAlive();

private:
    std::unique_ptr<msg_line> msg_req_ipc_;
    std::unique_ptr<msg_line> msg_resp_ipc_;
    std::atomic<int32_t> connected_;
    std::atomic<bool> stop_flag_;
    std::thread checkAvailable_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
