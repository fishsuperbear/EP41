#ifndef DOSOMEIP_SKELETON_H
#define DOSOMEIP_SKELETON_H

#include <mutex>
#include <stdint.h>
#include <functional>
#include "common/dosomeip_def.h"
#include "DoSomeIPStubImpl.hpp"


namespace hozon {
namespace netaos {
namespace diag {


class DoSomeIPSkeleton {
public:
    DoSomeIPSkeleton();
    virtual ~DoSomeIPSkeleton(){};

    // 建立通讯
    bool Init();
    void Deinit();
    void OfferService();
    void StopOfferService();

    void RegistUDSRequestCallback(std::function<void(const DoSomeIPReqUdsMessage&)> uds_request_callback);
    void UnRegistUDSRequestCallback();
    void RegistSomeIpLinkCallback(std::function<void(const DOSOMEIP_NETLINK_STATUS&)> someip_netlink_callback);
    void UnRegistSomeIpLinkCallback();

    // 当接收到UDS消息时候，主动调用回调函数，通知TPL。
    void OnReceiveDiagReq();
    // 当UDS消息处理完成，有应答/超时，调用此方法将result信息，返回给proxy端。
    void OnDiagProcessComplete(const DoSomeIPRespUdsMessage& resp);
    // 当客户端突然断联，需要通知TPL层
    void OnSomeIpDisConnect();

private:
    DoSomeIPSkeleton(const DoSomeIPSkeleton &);
    DoSomeIPSkeleton & operator = (const DoSomeIPSkeleton &);
private:
    std::shared_ptr<CommonAPI::Runtime> runtime_;
    std::shared_ptr<DoSomeIPStubImpl> stub_impl_;
    std::function<void(DoSomeIPReqUdsMessage)> uds_request_callback_;
    std::function<void(DOSOMEIP_NETLINK_STATUS )> someip_netlink_callback_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DOSOMEIP_SKELETON_H 