/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: 持久化模块的读操作代理
 * Created on: Feb 7, 2023
 *
 */
#ifndef MIDDLEWARE_PER_SRC_READ_ACCESSOR_IMPL_H_
#define MIDDLEWARE_PER_SRC_READ_ACCESSOR_IMPL_H_

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "include/basic_operations.h"
#include "include/read_accessor.h"
namespace hozon {
namespace netaos {
namespace per {
class ReadAccessorImpl : public ReadAccessor {
 public:
    ReadAccessorImpl() = default;
    virtual ~ReadAccessorImpl() = default;
    ReadAccessorImpl(const ReadAccessorImpl& obj) = delete;
    ReadAccessorImpl& operator=(const ReadAccessorImpl& obj) = delete;
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
    virtual std::fstream::int_type peek() noexcept;
    virtual std::fstream::int_type get() noexcept;
    virtual std::fstream::pos_type readbinary(const hozon::netaos::core::Span<char> s) noexcept;
    virtual std::fstream::pos_type readtext(std::string& s) noexcept;
    virtual ReadAccessor& readline(std::string& stream_string, const char delim = '\n') noexcept;
    bool open(const std::string& s, const BasicOperations::OpenMode mode) noexcept;
    void close() noexcept;

 private:
    std::ifstream fs_;
};
}  // namespace per
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_PER_SRC_READ_ACCESSOR_IMPL_H_
