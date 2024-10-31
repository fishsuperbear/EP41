
/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: KV文件存储接口
 * Created on: Feb 7, 2023
 *
 */
#include "include/key_value_storage.h"

#include <iostream>

#include "include/per_error_domain.h"
#include "src/file_recovery.h"
#include "src/json_key_value_storage_impl.h"
#include "src/proto_key_value_storage_impl.h"

namespace hozon {
namespace netaos {
namespace per {

template <class T>
hozon::netaos::core::Result<void> KeyValueStorage::SetBaseValue(const std::string& key, const T& value) noexcept {
    ProtoKeyValueStorageImpl* impl = dynamic_cast<ProtoKeyValueStorageImpl*>(this);
    JsonKeyValueStorageImpl* impl1 = dynamic_cast<JsonKeyValueStorageImpl*>(this);
    if (impl) {
        return impl->SetBaseValue(key, value);
    } else if (impl1) {
        return impl1->SetBaseValue(key, value);
    } else {
        return hozon::netaos::core::Result<void>::FromError(PerErrc::kKeyValueFormatUnsupported);
    }
}

template <class T>
hozon::netaos::core::Result<void> KeyValueStorage::GetBaseValue(const std::string& key, T& value) const noexcept {
    const ProtoKeyValueStorageImpl* impl = dynamic_cast<const ProtoKeyValueStorageImpl*>(this);
    const JsonKeyValueStorageImpl* impl1 = dynamic_cast<const JsonKeyValueStorageImpl*>(this);
    if (impl) {
        return impl->GetBaseValue(key, value);
    } else if (impl1) {
        return impl1->GetBaseValue(key, value);
    } else {
        return hozon::netaos::core::Result<void>::FromError(PerErrc::kKeyValueFormatUnsupported);
    }
}

// instantiate supportable tempate function.
template hozon::netaos::core::Result<void> KeyValueStorage::SetBaseValue<int>(const std::string& key, const int& value) noexcept;
template hozon::netaos::core::Result<void> KeyValueStorage::SetBaseValue<uint64_t>(const std::string& key, const uint64_t& value) noexcept;
template hozon::netaos::core::Result<void> KeyValueStorage::SetBaseValue<bool>(const std::string& key, const bool& value) noexcept;
template hozon::netaos::core::Result<void> KeyValueStorage::SetBaseValue<float>(const std::string& key, const float& value) noexcept;
template hozon::netaos::core::Result<void> KeyValueStorage::SetBaseValue<double>(const std::string& key, const double& value) noexcept;
template hozon::netaos::core::Result<void> KeyValueStorage::SetBaseValue<std::string>(const std::string& key, const std::string& value) noexcept;
template hozon::netaos::core::Result<void> KeyValueStorage::SetBaseValue<google::protobuf::Message>(const std::string& key, const google::protobuf::Message& value) noexcept;
template hozon::netaos::core::Result<void> KeyValueStorage::SetBaseValue<std::vector<uint8_t>>(const std::string& key, const std::vector<uint8_t>& value) noexcept;
template hozon::netaos::core::Result<void> KeyValueStorage::SetBaseValue<std::vector<int>>(const std::string& key, const std::vector<int>& value) noexcept;
template hozon::netaos::core::Result<void> KeyValueStorage::SetBaseValue<std::vector<bool>>(const std::string& key, const std::vector<bool>& value) noexcept;
template hozon::netaos::core::Result<void> KeyValueStorage::SetBaseValue<std::vector<float>>(const std::string& key, const std::vector<float>& value) noexcept;
template hozon::netaos::core::Result<void> KeyValueStorage::SetBaseValue<std::vector<double>>(const std::string& key, const std::vector<double>& value) noexcept;
template hozon::netaos::core::Result<void> KeyValueStorage::SetBaseValue<std::vector<std::string>>(const std::string& key, const std::vector<std::string>& value) noexcept;
template hozon::netaos::core::Result<void> KeyValueStorage::GetBaseValue<int>(const std::string& key, int& value) const noexcept;
template hozon::netaos::core::Result<void> KeyValueStorage::GetBaseValue<uint64_t>(const std::string& key, uint64_t& value) const noexcept;
template hozon::netaos::core::Result<void> KeyValueStorage::GetBaseValue<bool>(const std::string& key, bool& value) const noexcept;
template hozon::netaos::core::Result<void> KeyValueStorage::GetBaseValue<float>(const std::string& key, float& value) const noexcept;
template hozon::netaos::core::Result<void> KeyValueStorage::GetBaseValue<double>(const std::string& key, double& value) const noexcept;
template hozon::netaos::core::Result<void> KeyValueStorage::GetBaseValue<std::string>(const std::string& key, std::string& value) const noexcept;
template hozon::netaos::core::Result<void> KeyValueStorage::GetBaseValue<google::protobuf::Message>(const std::string& key, google::protobuf::Message& value) const noexcept;
template hozon::netaos::core::Result<void> KeyValueStorage::GetBaseValue<std::vector<uint8_t>>(const std::string& key, std::vector<uint8_t>& value) const noexcept;
template hozon::netaos::core::Result<void> KeyValueStorage::GetBaseValue<std::vector<int>>(const std::string& key, std::vector<int>& value) const noexcept;
template hozon::netaos::core::Result<void> KeyValueStorage::GetBaseValue<std::vector<bool>>(const std::string& key, std::vector<bool>& value) const noexcept;
template hozon::netaos::core::Result<void> KeyValueStorage::GetBaseValue<std::vector<float>>(const std::string& key, std::vector<float>& value) const noexcept;
template hozon::netaos::core::Result<void> KeyValueStorage::GetBaseValue<std::vector<double>>(const std::string& key, std::vector<double>& value) const noexcept;
template hozon::netaos::core::Result<void> KeyValueStorage::GetBaseValue<std::vector<std::string>>(const std::string& key, std::vector<std::string>& value) const noexcept;

hozon::netaos::core::Result<SharedHandle<KeyValueStorage>> OpenKeyValueStorage(const std::string& path, StorageConfig config) {
    PER_LOG_INFO << "OpenKeyValueStorage format:" << config.serialize_format;
    if (config.serialize_format == "proto") {
        std::shared_ptr<ProtoKeyValueStorageImpl> impl = std::make_shared<ProtoKeyValueStorageImpl>(config);
        if (!impl) {
            return hozon::netaos::core::Result<SharedHandle<KeyValueStorage>>::FromError(PerErrc::kOutOfMemoryError);
        }
        hozon::netaos::core::Result<void> result = impl->Open(path);
        if (!result.HasValue()) {
            return hozon::netaos::core::Result<SharedHandle<KeyValueStorage>>::FromError(result.Error());
        }
        return hozon::netaos::core::Result<SharedHandle<KeyValueStorage>>::FromValue(impl);
    } else if (config.serialize_format == "json") {
        std::shared_ptr<JsonKeyValueStorageImpl> impl = std::make_shared<JsonKeyValueStorageImpl>(config);
        if (!impl) {
            return hozon::netaos::core::Result<SharedHandle<KeyValueStorage>>::FromError(PerErrc::kOutOfMemoryError);
        }
        hozon::netaos::core::Result<void> result = impl->Open(path);
        if (!result.HasValue()) {
            return hozon::netaos::core::Result<SharedHandle<KeyValueStorage>>::FromError(result.Error());
        }
        return hozon::netaos::core::Result<SharedHandle<KeyValueStorage>>::FromValue(impl);
    } else {
        PER_LOG_INFO << "File format " << config.serialize_format << " is not supported.\n";
        return hozon::netaos::core::Result<SharedHandle<KeyValueStorage>>::FromError(PerErrc::kKeyValueFormatUnsupported);
    }
}

hozon::netaos::core::Result<void> RecoverKeyValueStorage(const std::string& path, StorageConfig config) {
    PER_LOG_INFO << "RecoverKeyValueStorage";
    FileRecovery recover;
    bool res = recover.RecoverHandle(path, config);
    if (!res) {
        return hozon::netaos::core::Result<void>::FromError(PerErrc::kKeyValueRecoveryFailed);
    }
    return hozon::netaos::core::Result<void>();
}

hozon::netaos::core::Result<void> ResetKeyValueStorage(const std::string& path, StorageConfig config) {
    PER_LOG_INFO << "ResetKeyValueStorage";
    FileRecovery recover;
    bool res = recover.ResetHandle(path, config);
    if (!res) {
        return hozon::netaos::core::Result<void>::FromError(PerErrc::kKeyValueResetFailed);
    }
    return hozon::netaos::core::Result<void>();
}
hozon::netaos::core::Result<void> IntegrityCheckKeyValueStorage(const std::string& path) {
    PER_LOG_INFO << "IntegrityCheckKeyValueStorage";
    FileRecovery recover;
    bool res = recover.CheckCrc32(path);
    if (!res) {
        return hozon::netaos::core::Result<void>::FromError(PerErrc::kIntegrityError);
    }
    return hozon::netaos::core::Result<void>();
}

}  // namespace per
}  // namespace netaos
}  // namespace hozon
