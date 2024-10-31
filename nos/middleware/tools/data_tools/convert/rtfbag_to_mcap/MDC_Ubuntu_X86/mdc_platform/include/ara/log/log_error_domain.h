/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: Defines the errors for log and trace.
 * Created: 2021-07-16
 */
#ifndef ARA_LOG_ERROR_DOMAIN_H
#define ARA_LOG_ERROR_DOMAIN_H
#include "ara/core/error_domain.h"
#include "ara/core/exception.h"

namespace ara {
namespace log {
enum class LogErrc : ara::core::ErrorDomain::CodeType {
    OPEN_FILE_FAIL = 1, // Error while opening the manifest file.
    KEY_NOT_FOUND = 2   // Key not found
};

class LogException : public ara::core::Exception {
public:
    explicit LogException(ara::core::ErrorCode errorCode) noexcept
        : ara::core::Exception(std::move(errorCode))
    {}
    ~LogException() override = default;
};

class LogErrorDomain final : public ara::core::ErrorDomain {
    /* Key ID for log and trace error domain. */
    constexpr static ErrorDomain::IdType kId {0x8000000000000102};

public:
    using Errc = LogErrc;
    using Exception = LogException;

    /* Creates a LogErrorDomain instance. */
    constexpr LogErrorDomain() noexcept
        : ErrorDomain(kId) {}
    ~LogErrorDomain() = default;

    const char* Name() const noexcept override
    {
        return "Log";
    }

    const char* Message(CodeType errorCode) const noexcept override
    {
        Errc const code = static_cast<Errc>(errorCode);
        switch (code) {
            case LogErrc::OPEN_FILE_FAIL:
                return "Failed to open the resource file";
            case LogErrc::KEY_NOT_FOUND:
                return "Requested key-value was not found";
            default:
                return "Unknown error";
        }
    }

    void ThrowAsException(const ara::core::ErrorCode &errorCode) const noexcept(false) override
    {
        ara::core::ThrowOrTerminate<Exception>(errorCode);
    }
};

namespace internal {
constexpr LogErrorDomain LOG_ERROR_DOMAIN;
}

inline constexpr const ara::core::ErrorDomain &GetLogDomain() noexcept
{
    return internal::LOG_ERROR_DOMAIN;
}

inline constexpr ara::core::ErrorCode MakeErrorCode(LogErrc code, ara::core::ErrorDomain::SupportDataType data) noexcept
{
    return ara::core::ErrorCode(static_cast<ara::core::ErrorDomain::CodeType>(code), GetLogDomain(), data);
}
}  // namespace log
}  // namespace ara
#endif  // ARA_LOG_ERROR_DOMAIN_H
