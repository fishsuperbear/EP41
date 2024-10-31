/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: 持久化模块的读操作
 * Created on: Feb 7, 2023
 *
 */
#ifndef MIDDLEWARE_PER_INCLUDE_READ_ACCESSOR_H_
#define MIDDLEWARE_PER_INCLUDE_READ_ACCESSOR_H_
#include <string>

#include "basic_operations.h"
#include "core/result.h"
#include "core/span.h"
#include "per_logger.h"

namespace hozon {
namespace netaos {
namespace per {
class ReadAccessor : public BasicOperations {
 public:
    ReadAccessor() = default;
    virtual ~ReadAccessor() = default;
    virtual std::fstream::int_type peek() noexcept = 0;
    virtual std::fstream::int_type get() noexcept = 0;
    virtual std::fstream::pos_type readbinary(const hozon::netaos::core::Span<char> s) noexcept = 0;
    virtual std::fstream::pos_type readtext(std::string& s) noexcept = 0;

    virtual ReadAccessor& readline(std::string& stream_string, char const delim = '\n') noexcept = 0;

 private:
    ReadAccessor(const ReadAccessor& obj) = delete;
    ReadAccessor& operator=(const ReadAccessor& obj) = delete;
};
}  // namespace per
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_PER_INCLUDE_READ_ACCESSOR_H_
