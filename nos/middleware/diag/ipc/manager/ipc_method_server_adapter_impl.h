#pragma once

#include "diag/ipc/common/ipc_def.h"
namespace hozon {
namespace netaos {
namespace diag {

class IPCMethodServerAdapterImpl {
public:
    IPCMethodServerAdapterImpl();
    ~IPCMethodServerAdapterImpl();
    int32_t Start(const std::string& service_name);
    int32_t Stop();
    using ProcessFunc = std::function<int32_t(const std::vector<uint8_t>&, std::vector<uint8_t>&)>;
    void RegisterProcess(ProcessFunc func);
    int32_t WaitRequest();
    
private:
    void CheckAlive();

private:
    ProcessFunc process_;
    std::unique_ptr<msg_line> msg_req_ipc_;
    std::unique_ptr<msg_line> msg_resp_ipc_;
    std::atomic<bool> connected_;
    std::atomic<bool> stop_flag_;
    std::thread checkAvailable_;
    std::thread waitReq_;
    std::atomic<bool> is_quit__ {false};
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
