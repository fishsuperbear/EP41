/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: This file provides an interface related to communication management.
 * Create: 2020-04-02
 */
#ifndef VRTF_VCC_API_DYNAMIC_ERROR_DOMAIN_H
#define VRTF_VCC_API_DYNAMIC_ERROR_DOMAIN_H
#include <map>
#include "ara/core/error_domain.h"
#include "ara/core/error_code.h"
#include "ara/core/exception.h"
#include "vrtf/vcc/api/types.h"

namespace vrtf {
namespace vcc {
namespace api {
namespace types {
enum class DynamicErrc: vrtf::core::ErrorDomain::CodeType {
    DEFAULT_ERRC = 0
};

class DynamicException : public ara::core::Exception {
public:
    explicit DynamicException(vrtf::core::ErrorCode && err) noexcept
        : Exception(std::move(err)) {}
    ~DynamicException(void) override = default;
};

class DynamicErrorDomain final : public vrtf::core::ErrorDomain {
public:
    using Errc = DynamicErrc;
    using Exception = DynamicException;

    explicit constexpr DynamicErrorDomain(const ErrorDomain::IdType& id = DEFAULT_DYNAMIC_ERROR_DOMAIN_ID) noexcept
        : ErrorDomain(id) {}
    ~DynamicErrorDomain(void) = default;
    char const *Name() const noexcept override
    {
        return "Dynamic";
    }
    char const *Message(vrtf::core::ErrorDomain::CodeType errorCode) const noexcept override
    {
        static_cast<void>(errorCode);
        return "Dynamic error domain doesn't support message output";
    }

    void ThrowAsException(vrtf::core::ErrorCode const &errorCode) const noexcept(false) override
    {
        ara::core::ThrowOrTerminate<Exception>(errorCode);
    }
private:
    // According to SWS_CORE_00016 Vendor-defined error domain range 0xc000’0000'0000'0000 ~ 0xc000’0000'ffff'ffff
    static constexpr  ErrorDomain::IdType DEFAULT_DYNAMIC_ERROR_DOMAIN_ID = 0xc000000000000001U;
};

constexpr DynamicErrorDomain DYNAMIC_ERROR_DOMAIN;

constexpr vrtf::core::ErrorDomain const &GetDynamicErrorDomain() noexcept
{
    return DYNAMIC_ERROR_DOMAIN;
}

constexpr vrtf::core::ErrorCode MakeErrorCode(DynamicErrc code,
                                              vrtf::core::ErrorDomain::SupportDataType data) noexcept
{
    return vrtf::core::ErrorCode(static_cast<vrtf::core::ErrorDomain::CodeType>(code),
                                 GetDynamicErrorDomain(), data);
}
}
}
}
}
#endif
