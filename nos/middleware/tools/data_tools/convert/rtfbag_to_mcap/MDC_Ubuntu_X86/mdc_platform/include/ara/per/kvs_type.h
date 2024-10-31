/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 * Description: kvstype.h
 * Create: 2019-12-10
 * Modify: 2020-06-10
 */

#ifndef ARA_PER_KVSTYPE_KVS_TYPE_H_
#define ARA_PER_KVSTYPE_KVS_TYPE_H_

#include <memory>
#include "ara/per/per_base_type.h"
#include "ara/per/per_error_domain.h"
namespace ara {
namespace per {
namespace kvstype {
class KvsType {
public:
    // @brief Status of the KVS access.
    enum class Status : uint8_t {
        kSuccess = 0U,    // kSuccess, indicates that the value was successfully restored from the KVS-storage.
        kNotFound,       // kNotfound, requested key was not found
        kIntegrityError, // kIntegrityError, the key-value pair was found, but the checksum of it is incorrect.
        kGeneralError    // kGeneralError, any other failure.
    };

    // @brief Supported types.
    enum class Type : uint8_t {
        kNotSupported = 0U,
        kFloat,
        kDouble,
        kSInt8,
        kSInt16,
        kSInt32,
        kSInt64,
        kUInt8,
        kUInt16,
        kUInt32,
        kUInt64,
        kString,
        kBinary,
        kBoolean,
        kObject,
        kNotSet
    };

    KvsType() noexcept(false);
    explicit KvsType(const bool& value) noexcept(false);
    explicit KvsType(const int8_t& value) noexcept(false);
    explicit KvsType(const int16_t& value) noexcept(false);
    explicit KvsType(const int32_t& value) noexcept(false);
    explicit KvsType(const int64_t& value) noexcept(false);
    explicit KvsType(const uint8_t& value) noexcept(false);
    explicit KvsType(const uint16_t& value) noexcept(false);
    explicit KvsType(const uint32_t& value) noexcept(false);
    explicit KvsType(const uint64_t& value) noexcept(false);
    explicit KvsType(const float32_t& value) noexcept(false);
    explicit KvsType(const float64_t& value) noexcept(false);
    explicit KvsType(const ara::core::String& value) noexcept(false);
    explicit KvsType(const char_t* value) noexcept(false);
    explicit KvsType(const void* const data, const std::size_t& len) noexcept(false);

    KvsType(const KvsType& src) noexcept(false);
    KvsType& operator=(const KvsType& src) noexcept(false);
    KvsType(KvsType&& src) noexcept(false);
    KvsType& operator=(KvsType&& src) noexcept(false);

    virtual ~KvsType();

    ara::core::Result<bool> GetKvsBool() const noexcept;

    ara::core::Result<int32_t> GetKvsSInt() const noexcept;

    ara::core::Result<int64_t> GetKvsSInt64() const noexcept;

    ara::core::Result<uint32_t> GetKvsUInt() const noexcept;

    ara::core::Result<uint64_t> GetKvsUInt64() const noexcept;

    ara::core::Result<float32_t> GetKvsFloat() const noexcept;

    ara::core::Result<float64_t> GetKvsDouble() const noexcept;

    ara::core::Result<ara::core::String> GetKvsString() const noexcept;

    ara::core::Result<ara::core::Vector<std::uint8_t>> GetKvsBinary() const noexcept;

    template <class T, typename std::enable_if<std::is_integral<T>::value, T>::type* = nullptr>
    ara::core::Result<ara::core::Vector<T>> GetKvsArray() const noexcept
    {
        using IntegralTypeResult = ara::core::Result<ara::core::Vector<T>>;
        if (GetType() == KvsType::Type::kNotSupported) {
            return IntegralTypeResult::FromError(ara::per::PerErrc::kDataTypeMismatchError);
        }
        ara::core::Vector<T> returnVector;
        returnVector.reserve(this->GetNumberOfArrayItems());
        ResetArrayItemGetter();
        for (std::size_t x = 0U; x < this->GetNumberOfArrayItems(); x++) {
            const auto next = this->GetNextArrayItem();
            if (!next.HasValue()) {
                return IntegralTypeResult::FromError(next.Error());
            }
            if (std::is_signed<T>::value) {
                const auto res = next.Value().GetKvsSInt64();
                if (!res.HasValue()) {
                    return IntegralTypeResult::FromError(res.Error());
                }
                returnVector.push_back(static_cast<T>(res.Value()));
            } else if (std::is_same<T, bool>::value) {
                const auto res = next.Value().GetKvsBool();
                if (!res.HasValue()) {
                    return IntegralTypeResult::FromError(res.Error());
                }
                returnVector.push_back(res.Value());
            } else {
                const auto res = next.Value().GetKvsUInt64();
                if (!res.HasValue()) {
                    return IntegralTypeResult::FromError(res.Error());
                }
                returnVector.push_back(static_cast<T>(res.Value()));
            }
        }
        return IntegralTypeResult::FromValue(std::move(returnVector));
    }

    template <class T, typename std::enable_if<std::is_floating_point<T>::value, T>::type* = nullptr>
    ara::core::Result<ara::core::Vector<T>> GetKvsArray() const noexcept
    {
        using FloatTypeResult = ara::core::Result<ara::core::Vector<T>>;
        if (GetType() == KvsType::Type::kNotSupported) {
            return FloatTypeResult::FromError(ara::per::PerErrc::kDataTypeMismatchError);
        }
        ara::core::Vector<T> returnVector;
        returnVector.reserve(this->GetNumberOfArrayItems());
        ResetArrayItemGetter();
        for (std::size_t x = 0U; x < this->GetNumberOfArrayItems(); x++) {
            const auto next = this->GetNextArrayItem();
            if (!next.HasValue()) {
                return FloatTypeResult::FromError(next.Error());
            }
            const auto res = next.Value().GetKvsDouble();
            if (!res.HasValue()) {
                return FloatTypeResult::FromError(res.Error());
            }
            returnVector.push_back(static_cast<T>(res.Value()));
        }
        return FloatTypeResult::FromValue(std::move(returnVector));
    }

    template <class T, typename std::enable_if<std::is_same<T, KvsType>::value, T>::type* = nullptr>
    ara::core::Result<ara::core::Vector<T>> GetKvsArray() const noexcept
    {
        using KvsTypeResult = ara::core::Result<ara::core::Vector<T>>;
        ara::core::Vector<T> returnVector;
        returnVector.reserve(this->GetNumberOfArrayItems());
        ResetArrayItemGetter();
        for (std::size_t x = 0U; x < this->GetNumberOfArrayItems(); x++) {
            auto next = this->GetNextArrayItem();
            if (!next.HasValue()) {
                return KvsTypeResult::FromError(next.Error());
            }
            returnVector.push_back(std::move(next).Value());
        }
        return KvsTypeResult::FromValue(std::move(returnVector));
    }

    template <class T, typename std::enable_if<std::is_same<T, ara::core::String>::value, T>::type* = nullptr>
    ara::core::Result<ara::core::Vector<T>> GetKvsArray() const noexcept
    {
        using StringTypeResult = ara::core::Result<ara::core::Vector<T>>;
        if (GetType() == KvsType::Type::kNotSupported) {
            return StringTypeResult::FromError(ara::per::PerErrc::kDataTypeMismatchError);
        }
        ara::core::Vector<T> returnVector;
        returnVector.reserve(this->GetNumberOfArrayItems());
        ResetArrayItemGetter();
        for (std::size_t x = 0U; x < this->GetNumberOfArrayItems(); x++) {
            const auto next = this->GetNextArrayItem();
            if (!next.HasValue()) {
                return StringTypeResult::FromError(next.Error());
            }
            const auto res = next.Value().GetKvsString();
            if (!res.HasValue()) {
                return StringTypeResult::FromError(res.Error());
            }
            returnVector.push_back(std::move(res).Value());
        }
        return StringTypeResult::FromValue(std::move(returnVector));
    }

    ara::core::Result<void> AddKvsArrayItem(const KvsType& kvs) const noexcept;

    ara::core::String GetKey() const noexcept;

    Status GetStatus() const noexcept;

    Type GetType() const noexcept;

    bool IsSignedInteger() const noexcept;

    bool IsUnsignedInteger() const noexcept;

    void SetKey(const ara::core::String& name) const noexcept;

    ara::core::String SerializeKvsData() const noexcept;

    std::uint64_t CalculateKvsTypeSize() const noexcept;

    std::size_t GetNumberOfArrayItems() const noexcept(false);

private:
    class Impl;

    std::unique_ptr<Impl> impl_ptr_;

    friend class InternalAccess;

    ara::core::Result<KvsType> GetNextArrayItem() const noexcept(false);

    void ResetArrayItemGetter() const noexcept(false);
};

template <typename... ArgTps>
KvsType CreateKvsType(ara::core::String const& key, ArgTps&&... arg)
{
    KvsType lResult(std::forward<ArgTps>(arg)...);
    lResult.SetKey(key);
    return lResult;
}
}  // namespace kvstype
}  // namespace per
}  // namespace ara

#endif  // ARA_PER_KVSTYPE_KVS_TYPE_H_
