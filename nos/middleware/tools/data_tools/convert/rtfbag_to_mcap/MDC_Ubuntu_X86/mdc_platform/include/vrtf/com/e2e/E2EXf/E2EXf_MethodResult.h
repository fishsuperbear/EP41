/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 * Description: Define Method result type
 * Create: 2021-07-20
 */
#ifndef VRTF_E2E_E2EXF_METHODRESULT_H
#define VRTF_E2E_E2EXF_METHODRESULT_H

/* AXIVION disable style AutosarC++19_03-A2.8.1 : it makes more sense to have these 2 ret classes in one single file */
#include <cstdint>
#include <vrtf/com/e2e/E2EXf/E2EXf_MethodType.h>
#include <vrtf/com/e2e/E2EXf/ResultImpl.h>
namespace vrtf {
namespace com {
namespace e2e {
namespace impl {
/* **************************** MethodProtectResult ********************** */
class MethodProtectResult final {
public:
    MethodProtectResult(ProtectResultImpl ProtectResult, std::uint32_t RequestCounter) : ProtectResult_(ProtectResult),
                                                                                         RequestCounter_(RequestCounter)
    {}

    MethodProtectResult(): MethodProtectResult(ProtectResultImpl::HardRuntimeError, 0U) {}

    ~MethodProtectResult() = default;

    MethodProtectResult(const MethodProtectResult& ClientProtectResultPre);

    MethodProtectResult& operator=(const MethodProtectResult& ClientProtectResultPre) &;

    /* AXIVION Next Line AutosarC++19_03-A3.1.6: this api is defined for external users */
    ProtectResultImpl GetProtectStatus() const noexcept;

    /* AXIVION Next Line AutosarC++19_03-A3.1.6: this api is defined for external users */
    std::uint32_t GetRequestCounter() const noexcept;

    /* AXIVION Next Line AutosarC++19_03-A3.1.6: this api is defined for external users */
    void SetRequestCounter(uint32_t requestCounter);

    /* AXIVION Next Line AutosarC++19_03-A3.1.6: this api is defined for external users */
    void SetProtectResult(ProtectResultImpl protectResult);

private:
    ProtectResultImpl ProtectResult_;
    std::uint32_t RequestCounter_;
};

/* **************************** MethodCheckResult ********************** */
class MethodCheckResult final {
public:
    MethodCheckResult(const CSTransactionHandleType &csTransactionHandle, const ResultImpl &ResultImpl,
                      uint32_t receivedRequestCounter)
            : CSTransactionHandle_(csTransactionHandle), ResultImpl_(ResultImpl),
              ReceivedRequestCounter_(receivedRequestCounter) {}

    MethodCheckResult() : MethodCheckResult(CSTransactionHandleType{0U, 0U},
                                            ResultImpl{ProfileCheckStatusImpl::kCheckDisabled,
                                                       SMStateImpl::kStateMDisabled}, 0U) {}
    ~MethodCheckResult() = default;
    MethodCheckResult(const MethodCheckResult& CheckResultPre);

    MethodCheckResult& operator=(const MethodCheckResult& CheckResultPre) & noexcept;

    /* AXIVION Next Line AutosarC++19_03-A3.1.6: this api is defined for external users */
    const ResultImpl &GetResultImpl() const;

    /* AXIVION Next Line AutosarC++19_03-A3.1.6: this api is defined for external users */
    const CSTransactionHandleType &GetCsTransactionHandle() const;

    /* AXIVION Next Line AutosarC++19_03-A3.1.6: this api is defined for external users */
    void SetCsTransactionHandle(const CSTransactionHandleType &csTransactionHandle);

    /* AXIVION Next Line AutosarC++19_03-A3.1.6: this api is defined for external users */
    uint32_t GetReceivedRequestCounter() const;

    void SetReceivedRequestCounter(uint32_t receivedRequestCounter);

private:
    CSTransactionHandleType CSTransactionHandle_;
    ResultImpl ResultImpl_;
    std::uint32_t ReceivedRequestCounter_;
};
} /* End namespace impl */
} /* End namesapce e2e */
} /* End namesapce com */
} /* End namespace vrtf */
/* AXIVION enable style AutosarC++19_03-A2.8.1 */
#endif // VRTF_E2E_E2EXF_METHODRESULT_H
