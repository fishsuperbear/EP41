
#include "api/dosomeip_transport.h"

namespace hozon {
namespace netaos {
namespace diag {

DoSomeIPTransport* DoSomeIPTransport::instance_ = nullptr;
std::mutex DoSomeIPTransport::mtx_;

DoSomeIPTransport* DoSomeIPTransport::getInstance() {
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new DoSomeIPTransport();
        }
    }
    return instance_;
}

DoSomeIPTransport::DoSomeIPTransport() {
    DoSomeIPLogger::GetInstance().CreateLogger("dosomeip");
    someip_mgr_ = std::make_unique<DoSomeIPManager>();
}

DOSOMEIP_RESULT
DoSomeIPTransport::DosomeipInit(std::function<void(const DoSomeIPReqUdsMessage&)> uds_request_callback, std::function<void(const DOSOMEIP_REGISTER_STATUS&)> someip_register_callback) {
    DS_INFO << "DoSomeIPTransport::DosomeipInit enter!";

    DOSOMEIP_RESULT res = someip_mgr_->Init(uds_request_callback, someip_register_callback);
    if (res == DOSOMEIP_RESULT::DOSOMEIP_RESULT_OK) {
        DS_INFO << "DoIPTransport::DosomeipInit finish!";
    } else {
        DS_ERROR << "DoIPTransport::DosomeipInit error!";
    }
    return res;
}

void DoSomeIPTransport::DosomeipDeinit() {
    DS_INFO << "DoSomeIPTransport::DosomeipDeinit";
    someip_mgr_->DeInit();

    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
}

DOSOMEIP_RESULT
DoSomeIPTransport::ReplyUdsOnSomeIp(const DoSomeIPRespUdsMessage& udsMsg) {
    DS_INFO << "DoSomeIPTransport::ReplyUdsOnSomeip enter!";
    auto res = someip_mgr_->DispatchUDSReply(udsMsg);
    if (res == DOSOMEIP_RESULT::DOSOMEIP_RESULT_OK) {
        DS_INFO << "DoIPTransport::ReplyUdsOnSomeip OK!";
    } else {
        DS_ERROR << "DoIPTransport::ReplyUdsOnSomeip error!";
    }
    return res;
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
   /* EOF */