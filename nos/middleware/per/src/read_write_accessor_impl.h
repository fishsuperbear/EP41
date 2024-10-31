/*
 * @Author: Z04975 liguoqiang@hozonauto.com
 * @Date: 2023-07-04 17:49:31
 * @LastEditors: Z04975 liguoqiang@hozonauto.com
 * @LastEditTime: 2023-09-13 17:55:13
 * @FilePath: /nos/middleware/per/src/read_write_accessor_impl.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: 持久化模块的读写操作代理
 * Created on: Feb 7, 2023
 *
 */
#ifndef MIDDLEWARE_PER_SRC_READ_WRITE_ACCESSOR_IMPL_H_
#define MIDDLEWARE_PER_SRC_READ_WRITE_ACCESSOR_IMPL_H_
#include <string>

#include "include/read_write_accessor.h"
#include "per_utils.h"
#include "src/file_recovery.h"
namespace hozon {

namespace netaos {
namespace per {
class ReadWriteAccessorImpl : public ReadWriteAccessor {
 public:
    explicit ReadWriteAccessorImpl(const StorageConfig& config);
    ReadWriteAccessorImpl() = default;
    ReadWriteAccessorImpl(const ReadWriteAccessorImpl& obj) = delete;
    ReadWriteAccessorImpl& operator=(const ReadWriteAccessorImpl& obj) = delete;
    virtual ~ReadWriteAccessorImpl();
    virtual std::fstream::pos_type tell() noexcept;
    virtual void seek(const std::fstream::pos_type pos) noexcept;
    virtual void seek(const std::fstream::off_type off, const BasicOperations::SeekDirection direction) noexcept;
    virtual bool good() const noexcept;
    virtual bool eof() const noexcept;
    virtual bool fail() const noexcept;
    virtual bool bad() const noexcept;
    virtual bool operator!() const noexcept;
    virtual explicit operator bool() const noexcept;
    virtual void clear() noexcept;
    virtual hozon::netaos::core::Result<void, int> fsync() noexcept;
    virtual std::fstream::pos_type writebinary(hozon::netaos::core::Span<char> s) noexcept;
    virtual std::fstream::pos_type writetext(const std::string& s) noexcept;
    virtual ReadWriteAccessor& operator<<(const std::string& stream_string) noexcept;
    virtual void flush() noexcept;
    virtual std::fstream::int_type peek() noexcept;
    virtual std::fstream::int_type get() noexcept;
    virtual std::fstream::pos_type readbinary(const hozon::netaos::core::Span<char> s) noexcept;
    virtual std::fstream::pos_type readtext(std::string& s) noexcept;
    virtual ReadWriteAccessor& readline(std::string& stream_string, const char delim = '\n') noexcept;
    bool open(const std::string& s, const BasicOperations::OpenMode mode) noexcept;
    void close() noexcept;

 private:
    std::fstream fs_;
    StorageConfig _config;
    std::string _filepath;
    FileRecovery* recover;
};
}  // namespace per
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_PER_SRC_READ_WRITE_ACCESSOR_IMPL_H_
