
#include "echo.h"
#include "impl/echo_impl.h"

namespace hozon {
namespace netaos {
namespace topic {

Echo::Echo() {
    echo_impl_ = std::make_unique<EchoImpl>();
}

Echo::~Echo() {
    if (echo_impl_) {
        echo_impl_ = nullptr;
    }
}

void Echo::Stop() {
    echo_impl_->Stop();
}

void Echo::Start(EchoOptions echo_options) {
    echo_impl_->Start(echo_options);
}
}  // namespace topic
}  //namespace netaos
}  //namespace hozon