/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: 持久化模块的错误类型
 * Created on: Feb 7, 2023
 *
 */
#ifndef MIDDLEWARE_PER_INCLUDE_PER_ERROR_DOMAIN_H_
#define MIDDLEWARE_PER_INCLUDE_PER_ERROR_DOMAIN_H_
#include <map>

#include "core/error_code.h"
#include "core/error_domain.h"
#include "core/exception.h"
#include "per_base_type.h"
namespace hozon {
namespace netaos {
namespace per {

enum PerErrc : hozon::netaos::core::ErrorDomain::CodeType {
    // The key was not found.
    kKeyNotFoundError,
    // A severe error which might happen during the operation,
    // such as out of memory or writing/reading to the storage return an error.
    kPhysicalStorageError,
    // The integrity of the storage could not be established.
    // This can happen when the structure of a key value database is corrupted, or a read-only file has no content.
    kIntegrityError,
    // The validation of redundancy measures failed for a single key, for the whole key value data base, or for a file.
    kValidationError,
    // The encryption or decryption failed for a single key, for the whole key value data base, or for a file.
    kEncryptionError,
    // The provided data type does not match the stored data type.
    kDataTypeMismatchError,
    // The operation could not be performed because no initial value is available.
    kInitValueNotAvailableError,
    // Undefined error, implementation specific.
    kInternalError,
    // The allocated storage quota was exceeded, or memory could not be allocated.
    kOutOfMemoryError,
    // The file was not found.
    kFileNotFoundError,
    // The path does not exist
    kPathNotFoundError,
    // The extracted target file does not exist
    kExtractTargetFileNotExistError,
    // file recover failed.
    kFileRecoveryFailed,
    // file delete failed.
    kFileDeleteError,
    // file reset failed.
    kFileResetError,
    // Read error
    kReadAccessError,
    // Write error
    kWriteAccessError,
    // Open error
    kOpenAccessError,
    // Key value file format error
    kKeyValueFormatError,
    // Key value file format is not supported.
    kKeyValueFormatUnsupported,
    // Key value does no exist.
    kKeyValueNone,
    // Key value recover failed.
    kKeyValueRecoveryFailed,
    // Key value reset failed.
    kKeyValueResetFailed,
    // serialize error
    kSerializeError,
    // deserialize error
    kDeSerializeError,
};

class PerErrorDomain : public hozon::netaos::core::ErrorDomain {
    static const ErrorDomain::IdType kId{101U};

 public:
    PerErrorDomain() noexcept : ErrorDomain(kId) {}

    char_t const* Name() const noexcept override { return "Per"; }

    char_t const* Message(CodeType errorCode) const noexcept override {
        const auto code = static_cast<PerErrc>(errorCode);
        const auto iter = errorMessageMap.find(code);
        return (iter != errorMessageMap.end()) ? (iter->second).c_str() : "null";
    }
    void ThrowAsException(hozon::netaos::core::ErrorCode const& errorCode) const noexcept(false) override { hozon::netaos::core::ThrowOrTerminate<hozon::netaos::core::Exception>(errorCode); }

    ~PerErrorDomain() {}

 private:
    std::map<PerErrc, hozon::netaos::core::String> errorMessageMap{};
};

namespace internal {
static const PerErrorDomain g_perErrorDomain;
}

inline constexpr hozon::netaos::core::ErrorDomain const& GetPerDomain() { return internal::g_perErrorDomain; }

inline constexpr hozon::netaos::core::ErrorCode MakeErrorCode(const PerErrc code, const hozon::netaos::core::ErrorDomain::SupportDataType dataType) {
    return hozon::netaos::core::ErrorCode(static_cast<hozon::netaos::core::ErrorDomain::CodeType>(code), GetPerDomain(), dataType);
}
}  // namespace per
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_PER_INCLUDE_PER_ERROR_DOMAIN_H_
