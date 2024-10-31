#pragma once

#include "diag/ipc/common/ipc_def.h"

namespace hozon {
namespace netaos {
namespace diag {

class IPCMethodClientAdapterImpl;
class IPCMethodClientAdapter {
public:
    explicit IPCMethodClientAdapter();
    ~IPCMethodClientAdapter();

    int32_t Init(const std::string& service_name);
    int32_t Deinit();
    int32_t Request(const std::vector<uint8_t>& req, std::vector<uint8_t>& resp, const int64_t& timeout_ms);
    int32_t RequestAndForget(const std::vector<uint8_t>& req);
    int32_t IsMatched();

private:
    std::unique_ptr<IPCMethodClientAdapterImpl> pimpl_;
};


}  // namespace diag
}  // namespace netaos
}  // namespace hozon
