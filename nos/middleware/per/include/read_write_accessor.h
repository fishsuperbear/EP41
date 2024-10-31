
/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: 持久化模块的读写操作
 * Created on: Feb 7, 2023
 *
 */
#ifndef MIDDLEWARE_PER_INCLUDE_READ_WRITE_ACCESSOR_H_
#define MIDDLEWARE_PER_INCLUDE_READ_WRITE_ACCESSOR_H_
#include <string>

#include "per_base_type.h"
#include "read_accessor.h"

namespace hozon {
namespace netaos {
namespace per {
class ReadWriteAccessor : public ReadAccessor {
 public:
    ReadWriteAccessor() = default;
    virtual ~ReadWriteAccessor() = default;
    virtual hozon::netaos::core::Result<void, int> fsync() noexcept = 0;
    virtual std::fstream::pos_type writebinary(hozon::netaos::core::Span<char_t> s) noexcept = 0;
    virtual std::fstream::pos_type writetext(const std::string& s) noexcept = 0;
    virtual ReadWriteAccessor& operator<<(const std::string& stream_string) noexcept = 0;
    virtual void flush() noexcept = 0;

 private:
    ReadWriteAccessor(const ReadWriteAccessor& obj) = delete;
    ReadWriteAccessor& operator=(const ReadWriteAccessor& obj) = delete;
};
}  // namespace per
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_PER_INCLUDE_READ_WRITE_ACCESSOR_H_
