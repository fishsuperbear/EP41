/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: 普通文件代理的接口
 * Created on: Feb 7, 2023
 *
 */
#ifndef MIDDLEWARE_PER_INCLUDE_FILE_STORAGE_H_
#define MIDDLEWARE_PER_INCLUDE_FILE_STORAGE_H_
#include <memory>
#include <string>

#include "basic_operations.h"
#include "json_object.h"
#include "per_base_type.h"
#include "read_write_accessor.h"
#include "shared_handle.h"
#include "unique_handle.h"

namespace hozon {
namespace netaos {
namespace per {
class FileStorage {
 public:
    FileStorage() = default;
    virtual ~FileStorage() = default;
    hozon::netaos::core::Result<void> DeleteFile() noexcept;
    hozon::netaos::core::Result<bool> FileExist() noexcept;
    hozon::netaos::core::Result<void> RecoverFile() noexcept;
    hozon::netaos::core::Result<void> ResetFile() noexcept;

    template <typename T>
    hozon::netaos::core::Result<std::string> SerializeObject(const T& data, bool tostyple = false) noexcept {
        return JsonObject::GetInstance().SerializeObject(data, tostyple);
    }
    template <typename T>
    hozon::netaos::core::Result<std::fstream::pos_type> WriteObject(const T& data, ReadWriteAccessor& accessor, bool tostyple = false) noexcept {
        hozon::netaos::core::Result<std::string> result = SerializeObject(data, tostyple);
        if (!result.HasValue()) {
            return hozon::netaos::core::Result<std::fstream::pos_type>::FromError(result.Error());
        }
        std::string strJson = result.Value();
        PER_LOG_INFO << "WriteObject:" << strJson.size();
        if (accessor) {
            accessor.writetext(strJson);
            accessor.fsync();
        } else {
            PER_LOG_ERROR << "accessor is null:";
            return hozon::netaos::core::Result<std::fstream::pos_type>::FromError(PerErrc::kOutOfMemoryError);
        }
        return hozon::netaos::core::Result<std::fstream::pos_type>::FromValue(strJson.size());
    }
    template <typename T>
    hozon::netaos::core::Result<T> DerializeObject(const std::string& data) noexcept {
        return JsonObject::GetInstance().DerializeObject<T>(data);
    }
    template <typename T>
    hozon::netaos::core::Result<std::fstream::pos_type> ReadObject(T& data, ReadAccessor& accessor) noexcept {
        std::string strJson;
        if (accessor) {
            accessor.readtext(strJson);
        } else {
            PER_LOG_ERROR << "accessor is null:";
            return hozon::netaos::core::Result<std::fstream::pos_type>::FromError(PerErrc::kOutOfMemoryError);
        }
        hozon::netaos::core::Result<T> result = DerializeObject<T>(strJson);
        if (!result.HasValue()) {
            return hozon::netaos::core::Result<std::fstream::pos_type>::FromError(result.Error());
        }
        data = result.Value();
        PER_LOG_INFO << "ReadObject:" << strJson.size();
        return hozon::netaos::core::Result<std::fstream::pos_type>::FromValue(strJson.size());
    }
    hozon::netaos::core::Result<UniqueHandle<ReadWriteAccessor>> OpenFileReadWrite(const BasicOperations::OpenMode mode) noexcept;
    hozon::netaos::core::Result<UniqueHandle<ReadAccessor>> OpenFileReadOnly(const BasicOperations::OpenMode mode) noexcept;
    hozon::netaos::core::Result<UniqueHandle<ReadWriteAccessor>> OpenFileWriteOnly(const BasicOperations::OpenMode mode) noexcept;

 private:
    FileStorage(const FileStorage& obj) = delete;
    FileStorage& operator=(const FileStorage& obj) = delete;
};

hozon::netaos::core::Result<SharedHandle<FileStorage>> OpenFileStorage(const std::string& filepath, StorageConfig config) noexcept;
hozon::netaos::core::Result<void> IntegrityCheckFileStorage(const std::string& path) noexcept;

}  // namespace per
}  // namespace netaos
}  // namespace hozon

#endif  // MIDDLEWARE_PER_INCLUDE_FILE_STORAGE_H_
