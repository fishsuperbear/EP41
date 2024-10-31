#include "diag/ipc/adapter/ipc_method_client_adapter.h"
#include "diag/ipc/manager/ipc_method_client_adapter_impl.h"

namespace hozon {
namespace netaos {
namespace diag {

IPCMethodClientAdapter::IPCMethodClientAdapter() :
pimpl_(std::make_unique<IPCMethodClientAdapterImpl>())
{}

IPCMethodClientAdapter::~IPCMethodClientAdapter() {}

int32_t IPCMethodClientAdapter::Init(const std::string& service_name) {
    return pimpl_->Init(service_name);
}

int32_t IPCMethodClientAdapter::Deinit() {
    return pimpl_->Deinit();
}

int32_t IPCMethodClientAdapter::Request(const std::vector<uint8_t>& req, std::vector<uint8_t>& resp, const int64_t& timeout_ms) {
    return pimpl_->Request(req, resp, timeout_ms);
}

int32_t IPCMethodClientAdapter::RequestAndForget(const std::vector<uint8_t>& req) {
    return pimpl_->RequestAndForget(req);
}

int32_t IPCMethodClientAdapter::IsMatched()
{
    return pimpl_->IsMatched();
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
