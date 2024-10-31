/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: 持久化模块的读写操作代理
 * Created on: Feb 7, 2023
 *
 */
#include "src/read_write_accessor_impl.h"

#include <utility>
#include <vector>

#include "include/read_write_accessor.h"

namespace hozon {
namespace netaos {
namespace per {
ReadWriteAccessorImpl::ReadWriteAccessorImpl(const StorageConfig& config) : _config(std::move(config)), recover(new FileRecovery()) {}
ReadWriteAccessorImpl::~ReadWriteAccessorImpl() { delete recover; }

std::fstream::pos_type ReadWriteAccessorImpl::tell() noexcept {
    PER_LOG_INFO << "tell()";
    return fs_.tellg();
}

void ReadWriteAccessorImpl::seek(std::fstream::pos_type const pos) noexcept {
    fs_.seekg(pos);
    PER_LOG_INFO << "seek()";
}

void ReadWriteAccessorImpl::seek(std::fstream::off_type const off, SeekDirection const direction) noexcept {
    fs_.seekg(off, std::ios_base::seekdir(direction));
    PER_LOG_INFO << "seek()";
}

std::fstream::int_type ReadWriteAccessorImpl::peek() noexcept {
    PER_LOG_INFO << "peek()";
    return fs_.peek();
}

std::fstream::int_type ReadWriteAccessorImpl::get() noexcept {
    PER_LOG_INFO << "get()";
    return fs_.get();
}

std::fstream::pos_type ReadWriteAccessorImpl::readbinary(const hozon::netaos::core::Span<char> s) noexcept {
    fs_.read(s.data(), s.size());
    std::streamsize read_count = fs_.gcount();
    fs_.seekg(0, std::istream::beg);
    PER_LOG_INFO << "readbinary() " << read_count;
    return read_count;
}
std::fstream::pos_type ReadWriteAccessorImpl::readtext(std::string& s) noexcept {
    std::ostringstream tmp;
    tmp << fs_.rdbuf();
    s = tmp.str();
    std::streamsize read_count = s.size();
    fs_.seekg(0, std::istream::beg);
    PER_LOG_INFO << "readtext() " << read_count;
    return read_count;
}

ReadWriteAccessor& ReadWriteAccessorImpl::readline(std::string& stream_string, char const delim /* = '\n' */) noexcept {
    std::vector<char> buf(40960);
    fs_.getline(buf.data(), buf.size(), delim);
    std::streamsize read_count = fs_.gcount();
    if (read_count > 0) {
        stream_string.assign(buf.data(), read_count);
    }
    PER_LOG_INFO << "readline() " << read_count;
    return *this;
}

bool ReadWriteAccessorImpl::good() const noexcept {
    PER_LOG_INFO << "good()";
    return fs_.good();
}

bool ReadWriteAccessorImpl::eof() const noexcept {
    PER_LOG_INFO << "eof()";
    return fs_.eof();
}

bool ReadWriteAccessorImpl::fail() const noexcept {
    PER_LOG_INFO << "fail()";
    return fs_.fail();
}

bool ReadWriteAccessorImpl::bad() const noexcept {
    PER_LOG_INFO << "bad()";
    return fs_.bad();
}

bool ReadWriteAccessorImpl::operator!() const noexcept { return (!fs_); }

ReadWriteAccessorImpl::operator bool() const noexcept { return (static_cast<bool>(fs_)); }

void ReadWriteAccessorImpl::clear() noexcept { fs_.clear(); }

hozon::netaos::core::Result<void, int> ReadWriteAccessorImpl::fsync() noexcept {
    int code = fs_.sync();
    recover->BackUpHandle(_filepath, _config);
    PER_LOG_INFO << "fsync()";
    return ((code == 0) ? hozon::netaos::core::Result<void, int>() : hozon::netaos::core::Result<void, int>(code));
}

std::fstream::pos_type ReadWriteAccessorImpl::writebinary(hozon::netaos::core::Span<char_t> s) noexcept {
    if (!PerUtils::CheckFreeSize(_filepath)) {
        return 0;
    }
    std::fstream::pos_type read_count = fs_.tellp();
    fs_.write(s.data(), s.size());
    read_count = fs_.tellp() - read_count;
    PER_LOG_INFO << "writebinary() " << read_count;
    return read_count;
}
std::fstream::pos_type ReadWriteAccessorImpl::writetext(const std::string& s) noexcept {
    if (!PerUtils::CheckFreeSize(_filepath)) {
        return 0;
    }
    std::fstream::pos_type read_count = fs_.tellp();
    fs_.write(s.c_str(), s.size());
    read_count = fs_.tellp() - read_count;
    PER_LOG_INFO << "writetext() " << read_count;
    return read_count;
}

bool ReadWriteAccessorImpl::open(const std::string& filepath, const BasicOperations::OpenMode mode) noexcept {
    fs_.open(filepath, std::ios_base::openmode(mode));
    _filepath = filepath;
    bool res = fs_.is_open();
    PER_LOG_INFO << "open() " << filepath << " res: " << res;
    return res;
}
void ReadWriteAccessorImpl::close() noexcept {
    fs_.close();
    PER_LOG_INFO << "close()";
}
void ReadWriteAccessorImpl::flush() noexcept {
    fs_.flush();
    PER_LOG_INFO << "flush()";
}

ReadWriteAccessor& ReadWriteAccessorImpl::operator<<(std::string const& stream_string) noexcept {
    fs_ << stream_string;
    return *this;
}

}  // namespace per
}  // namespace netaos
}  // namespace hozon
