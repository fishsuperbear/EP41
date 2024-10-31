/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: 文件基础操作
 * Created on: Feb 7, 2023
 *
 */
#ifndef MIDDLEWARE_PER_INCLUDE_BASIC_OPERATIONS_H_
#define MIDDLEWARE_PER_INCLUDE_BASIC_OPERATIONS_H_

#include <fstream>
// #include "per_base_type.h"
namespace hozon {
namespace netaos {
namespace per {
class BasicOperations {
 public:
    enum class SeekDirection : uint8_t {
        kBeg = 0U,  // Seek from the beginning.
        kCur = 1U,  // Seek from the current position.
        kEnd = 2U   // Seek from the end.
    };

    enum class OpenMode : uint8_t {
        kApp = 1L << 0,     // append to the end
        kAte = 1L << 1,     // put seek pointer at the end
        kBinary = 1L << 2,  // open as binary, will be opened as text if missing
        kIn = 1L << 3,
        kOut = 1L << 4,
        kTrunc = 1L << 5  // delete existing data at open
    };

    BasicOperations() = default;
    virtual ~BasicOperations() = default;

    virtual std::fstream::pos_type tell() noexcept = 0;

    virtual void seek(std::fstream::pos_type const pos) noexcept = 0;

    virtual void seek(std::fstream::off_type const off, SeekDirection const direction) noexcept = 0;

    virtual bool good() const noexcept = 0;

    virtual bool eof() const noexcept = 0;

    virtual bool fail() const noexcept = 0;

    virtual bool bad() const noexcept = 0;
    virtual bool operator!() const noexcept = 0;

    explicit virtual operator bool() const noexcept = 0;

    virtual void clear() noexcept = 0;

 private:
    BasicOperations(const BasicOperations& obj) = delete;
    BasicOperations& operator=(const BasicOperations& obj) = delete;
};

BasicOperations::OpenMode operator|(BasicOperations::OpenMode const& left, BasicOperations::OpenMode const& right) noexcept;

BasicOperations::OpenMode operator&(BasicOperations::OpenMode const& left, BasicOperations::OpenMode const& right) noexcept;
}  // namespace per
}  // namespace netaos
}  // namespace hozon

#endif  // MIDDLEWARE_PER_INCLUDE_BASIC_OPERATIONS_H_
