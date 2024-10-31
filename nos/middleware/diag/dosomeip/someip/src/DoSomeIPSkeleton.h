#ifndef DOSOMEIP_SKELETON_H
#define DOSOMEIP_SKELETON_H

#include <stdint.h>
#include <functional>
#include <mutex>
#include "diag/dosomeip/someip/src/DoSomeIPStubImpl.hpp"
#include "diag/dosomeip/common/dosomeip_def.h"

namespace hozon {
namespace netaos {
namespace diag {

class DoSomeIPSkeleton {
public:
    DoSomeIPSkeleton();
    virtual ~DoSomeIPSkeleton(){};

    // 建立通讯
    bool Init(const std::uint16_t& timeout);
    void Deinit();
    void OfferService(const std::int16_t& timeout);
    void StopOfferService();

    void RegistUDSRequestCallback(std::function<void(const DoSomeIPReqUdsMessage&)> uds_request_callback);
    void UnRegistUDSRequestCallback();
    void RegistSomeIpEstablishCallback(std::function<void(const DOSOMEIP_REGISTER_STATUS&)> someip_register_callback);
    void UnRegistSomeIpEstablishCallback();

    // 当接收到UDS消息时候，主动调用回调函数，通知TPL。
    void OnReceiveDiagReq(const DoSomeIPReqUdsMessage& req);
    // 当UDS消息处理完成，有应答/超时，调用此方法将result信息，返回给proxy端。
    bool OnDiagProcessComplete(const DoSomeIPRespUdsMessage& resp);

private:
    DoSomeIPSkeleton(const DoSomeIPSkeleton&);
    DoSomeIPSkeleton& operator=(const DoSomeIPSkeleton&);

private:
    std::shared_ptr<CommonAPI::Runtime> runtime_;
    std::shared_ptr<DoSomeIPStubImpl> stub_impl_;
    std::function<void(DoSomeIPReqUdsMessage)> uds_request_callback_;
    std::function<void(DOSOMEIP_REGISTER_STATUS)> someip_register_callback_;
    bool successfullyRegistered_;
    std::mutex mtx_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DOSOMEIP_SKELETON_H