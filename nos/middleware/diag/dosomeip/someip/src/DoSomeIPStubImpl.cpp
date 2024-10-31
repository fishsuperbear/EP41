// DoSomeIPStubImpl.cpp
#include "DoSomeIPStubImpl.hpp"
#include "diag/dosomeip/config/dosomeip_config.h"


DoSomeIPStubImpl::DoSomeIPStubImpl() : tpl_resp_() {}

DoSomeIPStubImpl::~DoSomeIPStubImpl() {}

using namespace hozon::netaos::diag;
namespace {
    const v1::commonapi::DoSomeIP::DoSomeIPRespUdsMessage emptyMessage{};
}  // namespace

void DoSomeIPStubImpl::udsMessageRequest(const std::shared_ptr<CommonAPI::ClientId> _client, v1::commonapi::DoSomeIP::DoSomeIPReqUdsMessage _req, udsMessageRequestReply_t _reply) {
    // 收到UDS请求
    DS_DEBUG << "DoSomeIPStubImpl::udsMessageRequest().";
    v1::commonapi::DoSomeIP::stdErrorTypeEnum methodError = v1::commonapi::DoSomeIP::stdErrorTypeEnum::NOT_OK;
    PrintRequest(_req);
    tpl_resp_ = emptyMessage;
    hozon::netaos::diag::DoSomeIPReqUdsMessage tmpMessage{};
    auto res = ConvertStruct(_req, tmpMessage);
    if (!res) {
        methodError = v1::commonapi::DoSomeIP::stdErrorTypeEnum::NOT_OK;
        DS_ERROR << "convert error.";
        _reply(methodError, emptyMessage);
        return;
    }
    DS_DEBUG << "convert success.";
    callback_(tmpMessage);

    // 设定超时机制，持续检查UDS消息是否为空（用于判断是否收到TPL层的回复）
    auto result = waitForStructOrTimeout(tpl_resp_);
    if (result == WaitResult::kTimeout) {
        DS_ERROR << "uds message process timeout.";
        methodError = v1::commonapi::DoSomeIP::stdErrorTypeEnum::NOT_OK;
        _reply(methodError, emptyMessage);
        return;
    }
    DS_DEBUG << "uds message process success.";
    methodError = v1::commonapi::DoSomeIP::stdErrorTypeEnum::OK;
    _reply(methodError, tpl_resp_);
}

void DoSomeIPStubImpl::RegistReceiveDiagReqCallback(std::function<void(const hozon::netaos::diag::DoSomeIPReqUdsMessage&)> uds_request_callback) {
    callback_ = uds_request_callback;
}

void DoSomeIPStubImpl::UnRegistReceiveDiagReqCallback() {
    callback_ = nullptr;
}

bool DoSomeIPStubImpl::SetRespFormTPL(const hozon::netaos::diag::DoSomeIPRespUdsMessage& resp) {
    std::unique_lock<std::mutex> lck(mtx_);
    DS_DEBUG << "DoSomeIPStubImpl::SetRespFormTPL().";
    PrintResponse(resp);
    auto res = ConvertStruct(resp, tpl_resp_);
    if (!res) {
        DS_ERROR << "convert error.";
        return false;
    }
    DS_DEBUG << "convert success.";
    condition_.notify_all();
    return true;
}

bool DoSomeIPStubImpl::isStructEmpty(const v1::commonapi::DoSomeIP::DoSomeIPRespUdsMessage& msg) {
    return msg == emptyMessage;
}

WaitResult DoSomeIPStubImpl::waitForStructOrTimeout(const v1::commonapi::DoSomeIP::DoSomeIPRespUdsMessage& msg) {
    DS_DEBUG << "waitForStructOrTimeout enter:";
    std::unique_lock<std::mutex> lock(mtx_);
    auto timeout = std::chrono::seconds(DoSomeIPConfig::Instance()->GetResponseMaxTimeout());

    // 循环检查结构体，直到它不为空或超时
    while (isStructEmpty(msg)) {
        if (std::cv_status::timeout == condition_.wait_for(lock, timeout)) {
            return WaitResult::kTimeout;
        }
    }
    return WaitResult::kSuccess;
}
