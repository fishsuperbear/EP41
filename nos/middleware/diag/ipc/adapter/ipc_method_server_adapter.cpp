#include "diag/ipc/adapter/ipc_method_server_adapter.h"
#include "diag/ipc/manager/ipc_method_server_adapter_impl.h"

namespace hozon {
namespace netaos {
namespace diag {


IPCMethodServerAdapter::IPCMethodServerAdapter() :
pimpl_(std::make_unique<IPCMethodServerAdapterImpl>())
{}

IPCMethodServerAdapter::~IPCMethodServerAdapter() {}

int32_t IPCMethodServerAdapter::Start(const std::string& service_name) {
    return pimpl_->Start(service_name);
}

int32_t IPCMethodServerAdapter::Stop() {
    return pimpl_->Stop();
}

void IPCMethodServerAdapter::RegisterProcess(ProcessFunc cb) {
    pimpl_->RegisterProcess(cb);
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
