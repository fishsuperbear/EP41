

/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: KV proto文件存储
 * Created on: Feb 7, 2023
 *
 */
#ifndef MIDDLEWARE_PROTO_PER_SRC_KEY_VALUE_STORAGE_IMPL_H_
#define MIDDLEWARE_PROTO_PER_SRC_KEY_VALUE_STORAGE_IMPL_H_

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "include/key_value_storage.h"
#include "src/file_recovery.h"
#include "src/per_inner_type.h"

namespace hozon {
namespace netaos {
namespace per {

class ProtoKeyValueStorageImpl : public KeyValueStorage {
 public:
    explicit ProtoKeyValueStorageImpl(const StorageConfig& config);
    ~ProtoKeyValueStorageImpl();
    ProtoKeyValueStorageImpl(const ProtoKeyValueStorageImpl& obj) = delete;
    ProtoKeyValueStorageImpl& operator=(const ProtoKeyValueStorageImpl& obj) = delete;
    virtual hozon::netaos::core::Result<std::vector<std::string>> GetAllKeys() const noexcept;
    virtual hozon::netaos::core::Result<bool> HasKey(const std::string& key) const noexcept;
    virtual hozon::netaos::core::Result<void> RemoveKey(const std::string& key) noexcept;
    virtual hozon::netaos::core::Result<void> RemoveAllKeys() noexcept;
    virtual hozon::netaos::core::Result<void> SyncToStorage() noexcept;
    template <class T>
    hozon::netaos::core::Result<void> GetValue(const std::string& key, T& value) const noexcept {
        return GetValueHelper(key, value);
    }
    template <class T>
    hozon::netaos::core::Result<void> SetValue(const std::string& key, const T& value) noexcept {
        return SetValueHelper(key, value);
    }

    hozon::netaos::core::Result<void> Open(const std::string& path);

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

    // template<typename T, EnableIfCustom<T>* = nullptr>
    // int64_t SetValueHelper(const std::string& key, const T& value)
    // {
    //     return SetCustomValue<T>(key, value);
    // }

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

 private:
    // void VectorToBcdString(const std::vector<uint8_t>& input, std::string& ouput);
    // void BcdStringToVector(const std::string& input, std::vector<uint8_t>& ouput);
    std::string path_;
    InnerKeyValueMap kv_map_;
    StorageConfig _config;
    FileRecovery* recover;
};

}  // namespace per
}  // namespace netaos
}  // namespace hozon

#endif  // MIDDLEWARE_PROTO_PER_SRC_KEY_VALUE_STORAGE_IMPL_H_
