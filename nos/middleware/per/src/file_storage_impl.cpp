/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: 普通文件代理
 * Created on: Feb 7, 2023
 *
 */
#include "src/file_storage_impl.h"

#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "include/kvs_type.h"
#include "include/per_error_domain.h"
#include "src/read_write_accessor_impl.h"

namespace hozon {
namespace netaos {
namespace per {
FileStorageImpl::FileStorageImpl(const std::string& fs, StorageConfig config) : _fs(fs), _config(std::move(config)), recover(new FileRecovery()) {}
FileStorageImpl::~FileStorageImpl() { delete recover; }

hozon::netaos::core::Result<void> FileStorageImpl::DeleteFile() noexcept {
    PER_LOG_INFO << "DeleteFile";
    if (recover->DeleteFile(_fs, _config)) {
        return hozon::netaos::core::Result<void>::FromError(PerErrc::kFileDeleteError);
    } else {
        return hozon::netaos::core::Result<void>();
    }
}
hozon::netaos::core::Result<bool> FileStorageImpl::FileExist() noexcept {
    PER_LOG_INFO << "FileExist";
    return recover->FileExist(_fs);
}
hozon::netaos::core::Result<void> FileStorageImpl::RecoverFile() noexcept {
    PER_LOG_INFO << "RecoverFile";
    if (recover->RecoverHandle(_fs, _config)) {
        return hozon::netaos::core::Result<void>();
    } else {
        return hozon::netaos::core::Result<void>::FromError(PerErrc::kFileRecoveryFailed);
    }
}
hozon::netaos::core::Result<void> FileStorageImpl::ResetFile() noexcept {
    PER_LOG_INFO << "ResetFile";
    if (recover->ResetHandle(_fs, _config)) {
        return hozon::netaos::core::Result<void>();
    } else {
        return hozon::netaos::core::Result<void>::FromError(PerErrc::kFileResetError);
    }
}
hozon::netaos::core::Result<UniqueHandle<ReadWriteAccessor>> FileStorageImpl::OpenFileReadWriteHelp(const BasicOperations::OpenMode mode) noexcept {
    PER_LOG_INFO << "OpenFileReadWrite";
    auto accessor = std::make_unique<ReadWriteAccessorImpl>(_config);
    if (!accessor) {
        PER_LOG_WARN << "accessor is nullptr  ";
        return hozon::netaos::core::Result<UniqueHandle<ReadWriteAccessor>>::FromError(PerErrc::kOutOfMemoryError);
        // return nullptr;
    }
    if (recover->RecoverHandle(_fs, _config)) {
        PER_LOG_INFO << "RecoverHandle  true";
    } else {
        PER_LOG_INFO << "RecoverHandle  false";
    }
    if (accessor->open(_fs, mode)) {
        return hozon::netaos::core::Result<UniqueHandle<ReadWriteAccessor>>::FromValue(std::move(accessor));
    } else {
        return hozon::netaos::core::Result<UniqueHandle<ReadWriteAccessor>>::FromError(PerErrc::kOpenAccessError);
    }
}
hozon::netaos::core::Result<UniqueHandle<ReadAccessor>> FileStorageImpl::OpenFileReadOnlyHelp(const BasicOperations::OpenMode mode) noexcept {
    PER_LOG_INFO << "OpenFileReadOnly";
    auto accessor = std::make_unique<ReadAccessorImpl>();
    if (!accessor) {
        PER_LOG_WARN << "accessor is nullptr  ";
        return hozon::netaos::core::Result<UniqueHandle<ReadAccessor>>::FromError(PerErrc::kOutOfMemoryError);
    }
    if (recover->RecoverHandle(_fs, _config)) {
        PER_LOG_INFO << "RecoverHandle  true";
    } else {
        PER_LOG_INFO << "RecoverHandle  false";
    }
    if (accessor->open(_fs, mode)) {
        return hozon::netaos::core::Result<UniqueHandle<ReadAccessor>>::FromValue(std::move(accessor));
    } else {
        return hozon::netaos::core::Result<UniqueHandle<ReadAccessor>>::FromError(PerErrc::kOpenAccessError);
    }
}
hozon::netaos::core::Result<UniqueHandle<ReadWriteAccessor>> FileStorageImpl::OpenFileWriteOnlyHelp(const BasicOperations::OpenMode mode) noexcept {
    PER_LOG_INFO << "OpenFileWriteOnlyHelp";
    auto accessor = std::make_unique<ReadWriteAccessorImpl>(_config);
    if (!accessor) {
        PER_LOG_WARN << "accessor is nullptr  ";
        return hozon::netaos::core::Result<UniqueHandle<ReadWriteAccessor>>::FromError(PerErrc::kOutOfMemoryError);
    }
    if (recover->RecoverHandle(_fs, _config)) {
        PER_LOG_INFO << "RecoverHandle  true";
    } else {
        PER_LOG_INFO << "RecoverHandle  false";
    }
    if (accessor->open(_fs, mode)) {
        return hozon::netaos::core::Result<UniqueHandle<ReadWriteAccessor>>::FromValue(std::move(accessor));
    } else {
        return hozon::netaos::core::Result<UniqueHandle<ReadWriteAccessor>>::FromError(PerErrc::kOpenAccessError);
    }
}

}  // namespace per
}  // namespace netaos
}  // namespace hozon
