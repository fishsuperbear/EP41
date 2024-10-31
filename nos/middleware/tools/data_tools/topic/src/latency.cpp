
#include "latency.h"
#include "impl/latency_impl.h"

namespace hozon {
namespace netaos {
namespace topic {

Latency::Latency() {
    latency_impl_ = std::make_unique<LatencyImpl>();
}

Latency::~Latency() {
    if (latency_impl_) {
        latency_impl_ = nullptr;
    }
}

void Latency::Stop() {
    latency_impl_->Stop();
}

void Latency::Start(LatencyOptions latency_options) {
    latency_impl_->Start(latency_options);
}
}  // namespace topic
}  //namespace netaos
}  //namespace hozon