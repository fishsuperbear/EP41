#pragma once

#include "diag/ipc/common/ipc_def.h"

namespace hozon {
namespace netaos {
namespace diag {

class IPCMethodServerAdapterImpl;
class IPCMethodServerAdapter {
public:
    IPCMethodServerAdapter();
    ~IPCMethodServerAdapter();

    int32_t Start(const std::string& service_name);
    int32_t Stop();
    using ProcessFunc = std::function<int32_t(const std::vector<uint8_t>&, std::vector<uint8_t>&)>;
    void RegisterProcess(ProcessFunc call);

private:
    std::unique_ptr<IPCMethodServerAdapterImpl> pimpl_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
