/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 * Description: KV文件存储接口
 * Create: 2019-6-26
 * Modify: 2020-6-12
 */

#ifndef ARA_PER_KEY_VALUE_STORAGE_H_
#define ARA_PER_KEY_VALUE_STORAGE_H_

#include <memory>
#include "ara/per/per_error_domain.h"
#include "ara/per/kvs_type.h"
#include "ara/per/unique_handle.h"
#include "ara/per/serializer.h"
#include "ara/per/serializer/internal/serializer.h"
#include "ara/per/serializer/internal/deserializer.h"
#include "ara/per/serializer/serializer.h"
#include "ara/per/serializer/deserializer.h"

namespace ara {
namespace per {
class KeyValueStorage {
public:
    KeyValueStorage() = default;
    virtual ~KeyValueStorage() = default;

    virtual ara::core::Result<ara::core::Vector<ara::core::String>> GetAllKeys() const noexcept = 0;
    virtual ara::core::Result<bool> HasKey(const ara::core::String& key) const noexcept = 0;
    virtual ara::core::Result<void> RemoveKey(const ara::core::StringView key) noexcept = 0;
    virtual ara::core::Result<void> RemoveAllKeys() noexcept = 0;
    virtual ara::core::Result<void> SyncToStorage() noexcept = 0;
    template <class T>
    ara::core::Result<void> GetValue(const ara::core::StringView& key, T& value) const noexcept
    {
        return GetValueHelper(key, value);
    }
    template <class T>
    ara::core::Result<void> SetValue(const ara::core::StringView& key, const T& value) noexcept
    {
        return SetValueHelper(key, value);
    }
    template <class T,
        typename std::enable_if<std::is_class<serialization::Serializer<T> >::value, T>::type* = nullptr>
    ara::core::Result<void> GetSerialValue(const ara::core::StringView key, T& value) const noexcept
    {
        const auto result = GetInternalValue(ara::core::String(key));
        if (result.HasValue()) {
            ara::per::kvstype::KvsType kvs = std::move(result).Value();
            ara::per::serialization::Serializer<T> ds;
            ds.KvsReader(kvs);
            ds.ReadProcess(value);
            return ara::core::Result<void>::FromValue();
        }
        return ara::core::Result<void>::FromError(ara::per::PerErrc::kKeyNotFoundError);
    }
    template <class T,
        typename std::enable_if<std::is_class<serialization::Serializer<T>>::value, T>::type* = nullptr>
    ara::core::Result<void> SetSerialValue(const ara::core::StringView key, const T& value) noexcept
    {
        ara::per::serialization::Serializer<T> serial;
        serial.KvsWriter(key);
        serial.WriteProcess(value);
        const ara::per::kvstype::KvsType kvs = serial.GetWriteValue();
        return SetInternalValue(ara::core::String(key), kvs);
    }

    template <class T,
        typename std::enable_if<std::is_class<kv_serialization::Serializer<T>>::value, T>::type* = nullptr>
    ara::core::Result<void> SetCustomValueInternal(const ara::core::StringView key, const T& value) noexcept
    {
        ara::per::kvstype::KvsType kv;
        try {
            const ara::per::kv_serialization::Serializer<T> serial(ara::core::String(key), value);
            serial.Serialize(kv);
        } catch (const std::exception &e) {
            PrintExceptionInfo((ara::core::String("Serialize data failed, message: ") + ara::core::String(e.what())));
            return ara::core::Result<void>::FromError(ara::per::PerErrc::kIntegrityError);
        } catch (...) {
            PrintExceptionInfo(ara::core::String("Catch a unknown exception in serializer."));
            return ara::core::Result<void>::FromError(ara::per::PerErrc::kPhysicalStorageError);
        }
        return SetInternalValue(ara::core::String(key), kv);
    }
    template <class T,
        typename std::enable_if<std::is_class<kv_serialization::Serializer<T>>::value, T>::type* = nullptr>
    ara::core::Result<void> GetCustomValueInternal(const ara::core::StringView key, T& value) const noexcept
    {
        const auto result = GetInternalValue(ara::core::String(key));
        if (!result.HasValue()) {
            return ara::core::Result<void>::FromError(std::move(result).Error());
        }
        const ara::per::kvstype::KvsType kv = std::move(result).Value();
        try {
            const ara::per::kv_serialization::Deserializer<T> deserial(kv);
            value = deserial.GetValue();
        } catch (const std::exception &e) {
            PrintExceptionInfo((ara::core::String("Deserializer data failed, message: ") + ara::core::String(e.what())));
            return ara::core::Result<void>::FromError(ara::per::PerErrc::kIntegrityError);
        } catch (...) {
            PrintExceptionInfo(ara::core::String("Catch a unknown exception in deserializer."));
            return ara::core::Result<void>::FromError(ara::per::PerErrc::kPhysicalStorageError);
        }
        return ara::core::Result<void>::FromValue();
    }
private:
    template <class T>
    ara::core::Result<void> SetBaseValue(const ara::core::StringView& key, const T& value) noexcept;

    template <class T>
    ara::core::Result<void> GetBaseValue(const ara::core::StringView& key, T& value) const noexcept;

    template<typename T, EnableIfBase<T>* = nullptr>
    ara::core::Result<void> SetValueHelper(const ara::core::StringView& key, const T& value)
    {
        return SetBaseValue<T>(key, value);
    }

    template<typename T, EnableIfBase<T>* = nullptr>
    ara::core::Result<void> SetValueHelper(const ara::core::StringView& key, const ara::core::Vector<T>& value)
    {
        return SetBaseValue<ara::core::Vector<T>>(key, value);
    }

    template<typename T, EnableIfCustom<T>* = nullptr>
    ara::core::Result<void> SetValueHelper(const ara::core::StringView& key, const T& value)
    {
        return SetCustomValue<T>(key, value);
    }

    template <class T, EnableIfBase<T>* = nullptr>
    ara::core::Result<void> GetValueHelper(const ara::core::StringView& key, T& value) const
    {
        return GetBaseValue<T>(key, value);
    }

    template <class T, EnableIfBase<T>* = nullptr>
    ara::core::Result<void> GetValueHelper(const ara::core::StringView& key, ara::core::Vector<T>& value) const
    {
        return GetBaseValue<ara::core::Vector<T>>(key, value);
    }

    template <class T, EnableIfCustom<T>* = nullptr>
    ara::core::Result<void> GetValueHelper(const ara::core::StringView& key, T& value) const
    {
        return GetCustomValue<T>(key, value);
    }

    template <class T>
    ara::core::Result<void> GetCustomValue(const ara::core::StringView& key, T& value) const noexcept
    {
        Container data;
        if (!ReadByteStream(ara::core::String(key), data)) {
            return ara::core::Result<void>::FromError(PerErrc::kKeyNotFoundError);
        }
        try {
            const bin_serialization::Deserializer<T> dataDeserializer(data.data(), data.size());
            value = dataDeserializer.GetValue();
            return ara::core::Result<void>::FromValue();
        } catch (std::runtime_error& e) {
            return ara::core::Result<void>::FromError(PerErrc::kDataTypeMismatchError);
        }
    }

    template <class T>
    ara::core::Result<void> SetCustomValue(const ara::core::StringView& key, const T& value) noexcept
    {
        Container data;
        try {
            const bin_serialization::Serializer<T> dataSerializer(value);
            dataSerializer.Serialize(data);
            return StoreByteStream(ara::core::String(key), data);
        } catch (std::runtime_error& e) {
            return ara::core::Result<void>::FromError(PerErrc::kDataTypeMismatchError);
        }
    }

protected:
    virtual ara::core::Result<void> ReadByteStream(const ara::core::String& key, Container& data) const noexcept
        = 0;
    virtual ara::core::Result<void> StoreByteStream(const ara::core::String& key, const Container& data) noexcept
        = 0;
    virtual ara::core::Result<ara::per::kvstype::KvsType> GetInternalValue(const ara::core::String& key) const = 0;
    virtual ara::core::Result<void> SetInternalValue(const ara::core::String& key,
       ara::per::kvstype::KvsType const& value) noexcept = 0;
    virtual void PrintExceptionInfo(const ara::core::String& info) const noexcept = 0;
    KeyValueStorage(const KeyValueStorage& obj) = delete;
    KeyValueStorage& operator=(const KeyValueStorage& obj) = delete;
};

ara::core::Result<ara::per::UniqueHandle<KeyValueStorage>> OpenKeyValueStorage(const ara::core::StringView& kvs)
    noexcept;
ara::core::Result<void> RecoverKeyValueStorage(const ara::core::StringView& kvs) noexcept;

ara::core::Result<void> ResetKeyValueStorage(const ara::core::StringView& kvs) noexcept;
}  // namespace per
}  // namespace ara
#endif  // ARA_PER_KEY_VALUE_STORAGE_H_
