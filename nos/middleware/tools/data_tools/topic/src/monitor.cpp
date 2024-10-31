
#include <monitor.h>
#include <impl/monitor_impl.h>
#include <monitor/cyber_topology_message.h>
#include <monitor/screen.h>
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "idl/generated/cm_protobufTypeObject.h"
#include "idl/generated/proto_methodPubSubTypes.h"
#include "idl/generated/proto_methodTypeObject.h"
#include "data_tools_logger.hpp"
#include "topic_manager.hpp"

namespace hozon {
namespace netaos {
namespace topic {

Monitor::Monitor() {
    monitor_impl_ = std::make_unique<MonitorImpl>();
}

Monitor::~Monitor() {
    if (monitor_impl_) {
        monitor_impl_ = nullptr;
    }
}

void Monitor::SigResizeHandle() {
    monitor_impl_->SigResizeHandle();
}

void Monitor::Start(MonitorOptions monitor_options) {
    monitor_impl_->Start(monitor_options);
    return;
}

void Monitor::Stop() {
    monitor_impl_->Stop();
}

}  // namespace topic
}  //namespace netaos
}  //namespace hozon