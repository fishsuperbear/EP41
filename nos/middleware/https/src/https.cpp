#include "https.h"
#include "https_impl.h"
#include "logger.h"

namespace hozon {
namespace netaos {
namespace https {

Https* Https::instance_ = nullptr;
static std::mutex inst_mutex_;

Https::~Https() {
    if (impl_) {
        delete impl_;
        impl_ = nullptr;
    }

    inited_ = false;
}

Https& Https::Instance() {
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(inst_mutex_);
        if (nullptr == instance_) {
            instance_ = new Https();
        }
    }
    return *instance_;
}

void Https::Destroy() {
    std::lock_guard<std::mutex> lck(inst_mutex_);
    if (instance_) {
        delete instance_;
        instance_ = nullptr;
    }
}

void Https::Init() { 
    if (!inited_ && !impl_) {
        impl_ = new HttpsImpl();
        impl_->Init();
        inited_ = true;
    }
}

int Https::HttpRequest(RequestPtr req_ptr, ResponseHandler handler) {
    if (impl_) {
        return impl_->HttpRequest(req_ptr, handler);
    }
    return 0;
}

bool Https::CancelRequest(int id) {
    if (impl_) {
        return impl_->CancelRequest(id);
    }
    return false;
}

bool Https::IsInited() {
    return inited_;
}


}  // namespace https
}  // namespace v2c
}  // namespace hozon