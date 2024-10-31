
#include "DoSomeIPSkeleton.h"

namespace hozon {
namespace netaos {
namespace diag {

DoSomeIPSkeleton::DoSomeIPSkeleton()
{
    runtime_ = CommonAPI::Runtime::get()
    stub_impl_ = std::make_shared<HelloWorldStubImpl>();
}

void 
DoSomeIPSkeleton::OfferService()
{
    DOSOMEIP_INFO << "DoSomeIPSkeleton::OfferService";
}

void 
DoSomeIPSkeleton::StopOfferService()
{
    DOSOMEIP_INFO << "DoSomeIPSkeleton::StopOfferService";
}

void 
DoSomeIPSkeleton::Init()
{
    OfferService();
    
    bool successfullyRegistered = runtime->runtime_->registerService("local", "test", stub_impl_);
    while (!successfullyRegistered) {
        DOSOMEIP_WARN << "Register Service failed, trying again in 100 milliseconds...";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        successfullyRegistered = runtime->runtime_->registerService("local", "test", stub_impl_);
    }
    // 连接建立，通知
    someip_netlink_callback_(DOSOMEIP_NETLINK_STATUS::DOSOMEIP_NETLINK_STATUS_UP);
}

void 
DoSomeIPSkeleton::Deinit()
{
    StopOfferService();
    runtime_->unregisterService("local", "test", stub_impl_);
}

void 
DoSomeIPSkeleton::RegistUDSRequestCallback(std::function<void(const DoSomeIPReqUdsMessage&)> uds_request_callback)
{
    uds_request_callback_ = uds_request_callback;
}

void DoSomeIPSkeleton::UnRegistUDSRequestCallback()
{
    uds_request_callback_ = nullptr;
}

void RegistSomeIpLinkCallback(std::function<void(const DOSOMEIP_NETLINK_STATUS&)> someip_netlink_callback)
{
    someip_netlink_callback_ = someip_netlink_callback;
}

void UnRegistSomeIpLinkCallback()
{
    someip_netlink_callback_ = nullptr;
}

void DoSomeIPSkeleton::OnReceiveDiagReq()
{
    // TODO:收到UDS消息，将这个传递给TPL 
    DoSomeIPReqUdsMessage req{};
    if (uds_request_callback_)
    {
        uds_request_callback_(req);
    }
    else
    {
        // do nothing
        DOSOMEIP_INFO << "uds_request_callback null, do not notify TPL.";
    }
}
    
void DoSomeIPSkeleton::OnDiagProcessComplete(DoSomeIPRespUdsMessage resp)
{
    // TODO:将诊断结果返回给客户端 resp
}

void DoSomeIPSkeleton::OnSomeIpDisConnect()
{
    // 连接断开，通知
    if (someip_netlink_callback_)
    {
        someip_netlink_callback_(DOSOMEIP_NETLINK_STATUS::DOSOMEIP_NETLINK_STATUS_DOWN);
    }
    else
    {
        // do nothing
        DOSOMEIP_INFO << "someip_netlink_callback null, do not notify TPL.";
    }
}



}  // namespace diag
}  // namespace netaos
}  // namespace hozon
/* EOF */