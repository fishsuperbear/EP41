#include "hz.h"
#include "impl/hz_impl.h"

namespace hozon {
namespace netaos {
namespace topic {

Hz::Hz() {
    hz_impl_ = std::make_unique<HzImpl>();
}

Hz::~Hz() {
    if (hz_impl_) {
        hz_impl_ = nullptr;
    }
}

void Hz::Stop() {
    hz_impl_->Stop();
}

void Hz::Start(HzOptions hz_options) {
    hz_impl_->Start(hz_options);
}

}  // namespace topic
}  //namespace netaos
}  //namespace hozon
