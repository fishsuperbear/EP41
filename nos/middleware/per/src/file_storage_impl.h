/*
 * @Author: Z04975 liguoqiang@hozonauto.com
 * @Date: 2023-06-01 19:59:47
 * @LastEditors: Z04975 liguoqiang@hozonauto.com
 * @LastEditTime: 2023-09-13 17:49:33
 * @FilePath: /nos/middleware/per/src/file_storage_impl.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: 普通文件代理
 * Created on: Feb 7, 2023
 *
 */

#ifndef MIDDLEWARE_PER_SRC_FILE_STORAGE_IMPL_H_
#define MIDDLEWARE_PER_SRC_FILE_STORAGE_IMPL_H_
#include <string>

#include "include/file_storage.h"
#include "src/file_recovery.h"
#include "src/read_accessor_impl.h"
namespace hozon {
namespace netaos {
namespace per {
class FileStorageImpl : public FileStorage {
 public:
    FileStorageImpl(const std::string& fs, StorageConfig config);
    ~FileStorageImpl();
    hozon::netaos::core::Result<void> DeleteFile() noexcept;
    hozon::netaos::core::Result<bool> FileExist() noexcept;
    hozon::netaos::core::Result<void> RecoverFile() noexcept;
    hozon::netaos::core::Result<void> ResetFile() noexcept;
    hozon::netaos::core::Result<UniqueHandle<ReadWriteAccessor>> OpenFileReadWriteHelp(const BasicOperations::OpenMode mode) noexcept;
    hozon::netaos::core::Result<UniqueHandle<ReadAccessor>> OpenFileReadOnlyHelp(const BasicOperations::OpenMode mode) noexcept;
    hozon::netaos::core::Result<UniqueHandle<ReadWriteAccessor>> OpenFileWriteOnlyHelp(const BasicOperations::OpenMode mode) noexcept;

 private:
    std::string _fs;
    StorageConfig _config;
    FileRecovery* recover;
};

}  // namespace per
}  // namespace netaos
}  // namespace hozon

#endif  // MIDDLEWARE_PER_SRC_FILE_STORAGE_IMPL_H_
