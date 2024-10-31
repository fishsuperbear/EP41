// DoSomeIPStubImpl.hpp
#ifndef DOSOMEIPSTUBIMPL_H_
#define DOSOMEIPSTUBIMPL_H_
#include <mutex>
#include <CommonAPI/CommonAPI.hpp>
#include <v1/commonapi/DoSomeIPStubDefault.hpp>
#include "diag/dosomeip/someip/src/DoSomeIPFuntions.h"
#include "diag/dosomeip/common/dosomeip_def.h"

enum class WaitResult {
    kSuccess,  // 结构体被成功赋值
    kTimeout   // 超时
};

class DoSomeIPStubImpl : public v1_0::commonapi::DoSomeIPStubDefault {
public:
    DoSomeIPStubImpl();
    virtual ~DoSomeIPStubImpl();
    virtual void udsMessageRequest(const std::shared_ptr<CommonAPI::ClientId> _client, v1::commonapi::DoSomeIP::DoSomeIPReqUdsMessage _req, udsMessageRequestReply_t _reply);

    void RegistReceiveDiagReqCallback(std::function<void(const hozon::netaos::diag::DoSomeIPReqUdsMessage&)> uds_request_callback);
    void UnRegistReceiveDiagReqCallback();
    bool SetRespFormTPL(const hozon::netaos::diag::DoSomeIPRespUdsMessage& resp);

private:
    WaitResult waitForStructOrTimeout(const v1::commonapi::DoSomeIP::DoSomeIPRespUdsMessage& msg);
    bool isStructEmpty(const v1::commonapi::DoSomeIP::DoSomeIPRespUdsMessage& msg);

private:
    std::function<void(hozon::netaos::diag::DoSomeIPReqUdsMessage)> callback_;
    v1::commonapi::DoSomeIP::DoSomeIPRespUdsMessage tpl_resp_;
    std::mutex mtx_;
    std::condition_variable condition_;
};
#endif /* DOSOMEIPSTUBIMPL_H_ */