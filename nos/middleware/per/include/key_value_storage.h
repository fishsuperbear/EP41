/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: KV文件存储接口
 * Created on: Feb 7, 2023
 *
 */
#ifndef MIDDLEWARE_PER_INCLUDE_KEY_VALUE_STORAGE_H_
#define MIDDLEWARE_PER_INCLUDE_KEY_VALUE_STORAGE_H_

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/result.h"
#include "core/span.h"
#include "json_object.h"
#include "per_base_type.h"
#include "serializer_type.h"
#include "shared_handle.h"

namespace hozon {
namespace netaos {
namespace per {

class KeyValueStorage {
 public:
    KeyValueStorage() = default;
    virtual ~KeyValueStorage() = default;
    virtual hozon::netaos::core::Result<std::vector<std::string>> GetAllKeys() const noexcept = 0;
    virtual hozon::netaos::core::Result<bool> HasKey(const std::string& key) const noexcept = 0;
    virtual hozon::netaos::core::Result<void> RemoveKey(const std::string& key) noexcept = 0;
    virtual hozon::netaos::core::Result<void> RemoveAllKeys() noexcept = 0;
    virtual hozon::netaos::core::Result<void> SyncToStorage() noexcept = 0;
    template <class T>
    hozon::netaos::core::Result<void> GetValue(const std::string& key, T& value) const noexcept {
        return GetValueHelper(key, value);
    }
    template <class T>
    hozon::netaos::core::Result<void> SetValue(const std::string& key, const T& value) noexcept {
        return SetValueHelper(key, value);
    }

 private:
    template <class T>
    hozon::netaos::core::Result<void> SetBaseValue(const std::string& key, const T& value) noexcept;

    template <class T>
    hozon::netaos::core::Result<void> GetBaseValue(const std::string& key, T& value) const noexcept;

    template <typename T, EnableIfBase<T>* = nullptr>
    hozon::netaos::core::Result<void> SetValueHelper(const std::string& key, const T& value) {
        return SetBaseValue<T>(key, value);
    }
    template <typename T, EnableIfBase<T>* = nullptr>
    hozon::netaos::core::Result<void> SetValueHelper(const std::string& key, const std::vector<T>& value) {
        return SetBaseValue<std::vector<T>>(key, value);
    }

    template <typename T, EnableIfCustom<T>* = nullptr>
    hozon::netaos::core::Result<void> SetValueHelper(const std::string& key, const T& value) {
        hozon::netaos::core::Result<std::string> result = JsonObject::GetInstance().SerializeObject(value, false);
        if (!result.HasValue()) {
            return hozon::netaos::core::Result<void>::FromError(result.Error());
        }
        std::string valstr = result.Value();
        return SetBaseValue<std::string>(key, valstr);
    }

    // template<typename T, EnableIfCustom<T>* = nullptr>
    // int64_t SetValueHelper(const std::string& key, const T& value)
    // {
    //     return SetCustomValue<T>(key, value);
    // }
    template <typename T, EnableIfCustom<T>* = nullptr>
    hozon::netaos::core::Result<void> GetValueHelper(const std::string& key, T& value) const {
        std::string valstr;
        auto res = GetBaseValue<std::string>(key, valstr);
        hozon::netaos::core::Result<T> result = JsonObject::GetInstance().DerializeObject<T>(valstr);
        if (!result.HasValue()) {
            return hozon::netaos::core::Result<void>::FromError(result.Error());
        }
        value = result.Value();
        return res;
    }
    template <class T, EnableIfBase<T>* = nullptr>
    hozon::netaos::core::Result<void> GetValueHelper(const std::string& key, T& value) const {
        return GetBaseValue<T>(key, value);
    }

    template <class T, EnableIfBase<T>* = nullptr>
    hozon::netaos::core::Result<void> GetValueHelper(const std::string& key, std::vector<T>& value) const {
        return GetBaseValue<std::vector<T>>(key, value);
    }

    // template <class T, EnableIfCustom<T>* = nullptr>
    // int64_t GetValueHelper(const std::string& key, T& value) const
    // {
    //     return GetCustomValue<T>(key, value);
    // }
};

hozon::netaos::core::Result<SharedHandle<KeyValueStorage>> OpenKeyValueStorage(const std::string& path, StorageConfig config);
hozon::netaos::core::Result<void> RecoverKeyValueStorage(const std::string& path, StorageConfig config);
hozon::netaos::core::Result<void> ResetKeyValueStorage(const std::string& path, StorageConfig config);
hozon::netaos::core::Result<void> IntegrityCheckKeyValueStorage(const std::string& path);

}  // namespace per
}  // namespace netaos
}  // namespace hozon

#endif  // MIDDLEWARE_PER_INCLUDE_KEY_VALUE_STORAGE_H_
