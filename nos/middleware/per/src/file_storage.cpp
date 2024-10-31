/*
 * @Author: Z04975 liguoqiang@hozonauto.com
 * @Date: 2023-07-04 17:49:31
 * @LastEditors: Z04975 liguoqiang@hozonauto.com
 * @LastEditTime: 2023-10-19 10:22:13
 * @FilePath: /nos/middleware/per/src/file_storage.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: 普通文件代理的接口
 * Created on: Feb 7, 2023
 *
 */
#include "include/file_storage.h"

#include "include/per_error_domain.h"
#include "src/file_recovery.h"
#include "src/file_storage_impl.h"

namespace hozon {
namespace netaos {
namespace per {
hozon::netaos::core::Result<void> FileStorage::DeleteFile() noexcept {
    FileStorageImpl* impl = dynamic_cast<FileStorageImpl*>(this);
    if (!impl) {
        return hozon::netaos::core::Result<void>::FromError(PerErrc::kOutOfMemoryError);
    } else {
        return impl->DeleteFile();
    }
}
hozon::netaos::core::Result<bool> FileStorage::FileExist() noexcept {
    FileStorageImpl* impl = dynamic_cast<FileStorageImpl*>(this);
    if (!impl) {
        return hozon::netaos::core::Result<bool>::FromError(PerErrc::kOutOfMemoryError);
    } else {
        return impl->FileExist();
    }
}
hozon::netaos::core::Result<void> FileStorage::RecoverFile() noexcept {
    FileStorageImpl* impl = dynamic_cast<FileStorageImpl*>(this);
    if (!impl) {
        return hozon::netaos::core::Result<void>::FromError(PerErrc::kOutOfMemoryError);
    } else {
        return impl->RecoverFile();
    }
}

hozon::netaos::core::Result<void> FileStorage::ResetFile() noexcept {
    FileStorageImpl* impl = dynamic_cast<FileStorageImpl*>(this);
    if (!impl) {
        return hozon::netaos::core::Result<void>::FromError(PerErrc::kOutOfMemoryError);
    } else {
        return impl->ResetFile();
    }
}
hozon::netaos::core::Result<UniqueHandle<ReadWriteAccessor>> FileStorage::OpenFileReadWrite(const BasicOperations::OpenMode mode) noexcept {
    FileStorageImpl* impl = dynamic_cast<FileStorageImpl*>(this);
    if (!impl) {
        return hozon::netaos::core::Result<UniqueHandle<ReadWriteAccessor>>::FromError(PerErrc::kOutOfMemoryError);
    } else {
        return impl->OpenFileReadWriteHelp(mode);
    }
}
hozon::netaos::core::Result<UniqueHandle<ReadAccessor>> FileStorage::OpenFileReadOnly(const BasicOperations::OpenMode mode) noexcept {
    FileStorageImpl* impl = dynamic_cast<FileStorageImpl*>(this);
    if (!impl) {
        return hozon::netaos::core::Result<UniqueHandle<ReadAccessor>>::FromError(PerErrc::kOutOfMemoryError);
    } else {
        return impl->OpenFileReadOnlyHelp(mode);
    }
}
hozon::netaos::core::Result<UniqueHandle<ReadWriteAccessor>> FileStorage::OpenFileWriteOnly(const BasicOperations::OpenMode mode) noexcept {
    FileStorageImpl* impl = dynamic_cast<FileStorageImpl*>(this);
    if (!impl) {
        return hozon::netaos::core::Result<UniqueHandle<ReadWriteAccessor>>::FromError(PerErrc::kOutOfMemoryError);
    } else {
        return impl->OpenFileWriteOnlyHelp(mode);
    }
}

hozon::netaos::core::Result<SharedHandle<FileStorage>> OpenFileStorage(const std::string& filepath, StorageConfig config) noexcept {
    std::shared_ptr<FileStorageImpl> impl = std::make_shared<FileStorageImpl>(filepath, config);
    if (!impl) {
        return hozon::netaos::core::Result<SharedHandle<FileStorage>>::FromError(PerErrc::kOutOfMemoryError);
    } else {
        return hozon::netaos::core::Result<SharedHandle<FileStorage>>::FromValue(impl);
    }
}
hozon::netaos::core::Result<void> IntegrityCheckFileStorage(const std::string& path) noexcept {
    FileRecovery recover;
    bool res = recover.CheckCrc32(path);
    if (!res) {
        return hozon::netaos::core::Result<void>::FromError(PerErrc::kIntegrityError);
    } else {
        return hozon::netaos::core::Result<void>();
    }
}
}  // namespace per
}  // namespace netaos
}  // namespace hozon
