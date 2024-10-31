/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: This file provides an interface related to communication management.
 * Create: 2020-04-02
 */
#ifndef VCC_COM_ERROR_DOMAIN_H
#define VCC_COM_ERROR_DOMAIN_H
#include <map>
#include "ara/core/error_domain.h"
#include "ara/core/error_code.h"
#include "ara/core/exception.h"
#include "vrtf/vcc/api/types.h"

namespace vrtf {
namespace vcc {
namespace api {
namespace types {
enum class VccErrc: vrtf::core::ErrorDomain::CodeType {
    OK = 0,
    ERROR = 1,
    INVALID_ARGUMENT = 2,
    WRONG_DISTRIBUTED_CONFIG = 3, // not support
    JSON_PARSER_FAILED = 4,
    ENTITY_NOT_EXIST = 5
};

class VccException : public ara::core::Exception {
public:
    explicit VccException(vrtf::core::ErrorCode && err) noexcept
        : Exception(std::move(err)) {}
    ~VccException(void) override = default;
};

class VccErrorDomain final : public vrtf::core::ErrorDomain {
public:
    using Errc = VccErrc;
    using Exception = VccException;

    constexpr VccErrorDomain() noexcept
        : ErrorDomain(VCC_ERROR_DOMAIN_ID) {}
    ~VccErrorDomain(void) = default;
    char const *Name() const noexcept override
    {
        return "Vcc";
    }
    char const *Message(vrtf::core::ErrorDomain::CodeType errorCode) const noexcept override
    {
        static std::map<vrtf::core::ErrorDomain::CodeType, const std::string> mapCode{
            {vrtf::core::ErrorDomain::CodeType(VccErrc::OK), "OK"},
            {vrtf::core::ErrorDomain::CodeType(VccErrc::ERROR), "ERROR"},
            {vrtf::core::ErrorDomain::CodeType(VccErrc::INVALID_ARGUMENT), "INVALID_ARGUMENT"},
            {vrtf::core::ErrorDomain::CodeType(VccErrc::WRONG_DISTRIBUTED_CONFIG), "WRONG_DISTRIBUTED_CONFIG"},
            {vrtf::core::ErrorDomain::CodeType(VccErrc::JSON_PARSER_FAILED), "JSON_PARSER_FAILED"},
            {vrtf::core::ErrorDomain::CodeType(VccErrc::ENTITY_NOT_EXIST), "ENTITY_NOT_EXIST"}
        };
            return mapCode[errorCode].c_str();
    }

    void ThrowAsException(vrtf::core::ErrorCode const &errorCode) const noexcept(false) override
    {
        ara::core::ThrowOrTerminate<Exception>(errorCode);
    }
private:
    constexpr static  ErrorDomain::IdType VCC_ERROR_DOMAIN_ID = 0x8000000000001267U;
};

constexpr VccErrorDomain VCC_ERROR_DOMAIN;

constexpr vrtf::core::ErrorDomain const &GetVccErrorDomain() noexcept
{
    return VCC_ERROR_DOMAIN;
}

constexpr vrtf::core::ErrorCode MakeErrorCode(VccErrc code,
                                              vrtf::core::ErrorDomain::SupportDataType data) noexcept
{
    return vrtf::core::ErrorCode(static_cast<vrtf::core::ErrorDomain::CodeType>(code),
                                 GetVccErrorDomain(), data);
}
}
}
}
}
#endif
