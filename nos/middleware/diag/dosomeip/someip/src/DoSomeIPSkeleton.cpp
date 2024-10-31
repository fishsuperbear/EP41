
#include "DoSomeIPSkeleton.h"

namespace hozon {
namespace netaos {
namespace diag {

const std::string DOMAIN = "local";
const std::string INSTANCE = "commonapi.dosomeip";
const std::string CONNECTION = "service-sample";
const std::string INTERFACE = "commonapi.DoSomeIP:v1_0";

DoSomeIPSkeleton::DoSomeIPSkeleton() : successfullyRegistered_(false) {
    runtime_ = CommonAPI::Runtime::get();
    stub_impl_ = std::make_shared<DoSomeIPStubImpl>();
}

void DoSomeIPSkeleton::OfferService(const std::int16_t& timeout) {
    std::lock_guard<std::mutex> lck(mtx_);
    DS_INFO << "DoSomeIPSkeleton::OfferService";
    int16_t register_counts = timeout;
    std::thread registerTD([&]() {
        while (!successfullyRegistered_ ) {
            if(register_counts < 0)
            {
                break;
            }
            DS_WARN << "Register Service failed, trying again in 1 seconds...";
            std::this_thread::sleep_for(std::chrono::seconds(1));
            --register_counts;
            successfullyRegistered_ = runtime_->registerService(DOMAIN, INSTANCE, stub_impl_, CONNECTION);
            if (successfullyRegistered_) {
                DS_DEBUG << "DoSomeIPSkeleton::register service success!";
                someip_register_callback_(DOSOMEIP_REGISTER_STATUS::DOSOMEIP_REGISTER_STATUS_OK);
                break;
            }
        }
        if (!successfullyRegistered_) {
            DS_DEBUG << "DoSomeIPSkeleton::register service timeout!";
            someip_register_callback_(DOSOMEIP_REGISTER_STATUS::DOSOMEIP_REGISTER_STATUS_TIMEOUT);
        }
        DS_INFO << "Register Service successful!";
    });
    registerTD.detach();
}

void DoSomeIPSkeleton::StopOfferService() {
    DS_INFO << "DoSomeIPSkeleton::StopOfferService";
    runtime_->unregisterService(DOMAIN, INTERFACE, INSTANCE);
}

bool DoSomeIPSkeleton::Init(const std::uint16_t& timeout) {
    DS_INFO << "DoSomeIPSkeleton::Init";
    stub_impl_->RegistReceiveDiagReqCallback(std::bind(&DoSomeIPSkeleton::OnReceiveDiagReq, this, std::placeholders::_1));

    OfferService(timeout);
    return true;
}

void DoSomeIPSkeleton::Deinit() {
    DS_INFO << "DoSomeIPSkeleton::Deinit";
    StopOfferService();
}

void DoSomeIPSkeleton::RegistUDSRequestCallback(std::function<void(const DoSomeIPReqUdsMessage&)> uds_request_callback) {
    uds_request_callback_ = uds_request_callback;
}

void DoSomeIPSkeleton::UnRegistUDSRequestCallback() {
    uds_request_callback_ = nullptr;
}

void DoSomeIPSkeleton::RegistSomeIpEstablishCallback(std::function<void(const DOSOMEIP_REGISTER_STATUS&)> someip_register_callback) {
    someip_register_callback_ = someip_register_callback;
}

void DoSomeIPSkeleton::UnRegistSomeIpEstablishCallback() {
    someip_register_callback_ = nullptr;
}

void DoSomeIPSkeleton::OnReceiveDiagReq(const DoSomeIPReqUdsMessage& req) {
    DS_INFO << "DoSomeIPSkeleton::OnReceiveDiagReq";

    if (!successfullyRegistered_) {
        DS_ERROR << "connect error!";
        return;
    }

    // 收到UDS消息，将这个传递给TPL
    if (uds_request_callback_) {
        uds_request_callback_(req);
    } else {
        // do nothing
        DS_INFO << "uds_request_callback null, do not notify TPL.";
    }
}

bool DoSomeIPSkeleton::OnDiagProcessComplete(const DoSomeIPRespUdsMessage& resp) {
    DS_INFO << "DoSomeIPSkeleton::OnDiagProcessComplete";
    if (!successfullyRegistered_) {
        DS_ERROR << "connect error!";
        return false;
    }

    // 将诊断结果返回给客户端 resp
    auto res = stub_impl_->SetRespFormTPL(resp);
    if (res) {
        DS_DEBUG << "SetRespFormTPL success!";
    } else {
        DS_ERROR << "SetRespFormTPL failed!";
        return false;
    }

    return true;
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
   /* EOF */