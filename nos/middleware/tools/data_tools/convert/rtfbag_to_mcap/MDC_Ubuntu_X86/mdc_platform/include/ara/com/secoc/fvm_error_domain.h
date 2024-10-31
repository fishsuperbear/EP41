/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
 * Description: This file provides error code of fvm.
 * Create: 2021-11-07
 */
#ifndef ARA_COM_SECOC_FVM_ERROR_DOMAIN_H
#define ARA_COM_SECOC_FVM_ERROR_DOMAIN_H
#include <map>
#include "ara/core/error_domain.h"
#include "ara/core/error_code.h"
#include "ara/core/exception.h"
#include "vrtf/vcc/api/types.h"
namespace ara {
namespace com {
namespace secoc {
enum class SecOcFvmErrc : ara::core::ErrorDomain::CodeType {
    kFVNotAvailable = 1,
    kFVInitializeFailed = 2
};

class SecOcFvmException : public ara::core::Exception {
public:
    explicit SecOcFvmException(ara::core::ErrorCode && err) noexcept
        : Exception(std::move(err)) {}
    ~SecOcFvmException(void) = default;
};

class SecOcFvmErrorDomain final : public ara::core::ErrorDomain {
public:
    using Errc = SecOcFvmErrc;
    using Exception = SecOcFvmException;

    static constexpr ErrorDomain::IdType SecOcFvmErrorDomainId = 0x8000000000001271U; // SWS_CM_11341

    constexpr SecOcFvmErrorDomain() noexcept
        : ErrorDomain(SecOcFvmErrorDomainId) {}
    ~SecOcFvmErrorDomain(void) = default;
    char const *Name() const noexcept override
    {
        return "SecOcFvm";
    }

    char const *Message(ara::core::ErrorDomain::CodeType errorCode) const noexcept override
    {
        switch (static_cast<Errc>(errorCode)) {
            case Errc::kFVNotAvailable:
                return "Freshness Value not available.";
            case Errc::kFVInitializeFailed:
                return "Freshness Value Manager could not be used.";
            default:
                ara::core::Abort("Unknown SecOcFvm error");
                return "Unknown SecOcFvm error.";
        }
    }

    void ThrowAsException(ara::core::ErrorCode const &errorCode) const noexcept(false) override
    {
        ara::core::ThrowOrTerminate<Exception>(errorCode);
    }
};

constexpr SecOcFvmErrorDomain SECOCFVM_ERROR_DOMAIN;

constexpr ara::core::ErrorDomain const &GetSecOcFvmErrorDomain() noexcept
{
    return SECOCFVM_ERROR_DOMAIN;
}

constexpr ara::core::ErrorCode MakeErrorCode(SecOcFvmErrc code,
                                             ara::core::ErrorDomain::SupportDataType data) noexcept
{
    return vrtf::core::ErrorCode(static_cast<vrtf::core::ErrorDomain::CodeType>(code),
                                 GetSecOcFvmErrorDomain(), data);
}
}
}
}
#endif  // ARA_COM_SECOC_FVM_ERROR_DOMAIN_H
