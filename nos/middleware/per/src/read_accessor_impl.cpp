/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: 持久化模块的读操作代理
 * Created on: Feb 7, 2023
 *
 */
#include "src/read_accessor_impl.h"

#include <vector>

namespace hozon {
namespace netaos {
namespace per {

std::fstream::pos_type ReadAccessorImpl::tell() noexcept { return fs_.tellg(); }

void ReadAccessorImpl::seek(std::fstream::pos_type const pos) noexcept {
    fs_.seekg(pos);
    PER_LOG_INFO << "seek()";
}

void ReadAccessorImpl::seek(std::fstream::off_type const off, SeekDirection const direction) noexcept {
    fs_.seekg(off, std::ios_base::seekdir(direction));
    PER_LOG_INFO << "seek()";
}

bool ReadAccessorImpl::good() const noexcept {
    PER_LOG_INFO << "good()";
    return fs_.good();
}

bool ReadAccessorImpl::eof() const noexcept {
    PER_LOG_INFO << "eof()";
    return fs_.eof();
}

bool ReadAccessorImpl::fail() const noexcept {
    PER_LOG_INFO << "fail()";
    return fs_.fail();
}

bool ReadAccessorImpl::bad() const noexcept {
    PER_LOG_INFO << "bad()";
    return fs_.bad();
}

bool ReadAccessorImpl::operator!() const noexcept { return (!fs_); }

ReadAccessorImpl::operator bool() const noexcept { return (static_cast<bool>(fs_)); }

void ReadAccessorImpl::clear() noexcept {
    fs_.clear();
    PER_LOG_INFO << "clear()";
}
void ReadAccessorImpl::close() noexcept {
    fs_.close();
    PER_LOG_INFO << "close()";
}

std::fstream::int_type ReadAccessorImpl::peek() noexcept {
    PER_LOG_INFO << "peek()";
    return fs_.peek();
}

std::fstream::int_type ReadAccessorImpl::get() noexcept {
    PER_LOG_INFO << "get()";
    return fs_.get();
}

bool ReadAccessorImpl::open(const std::string& s, const OpenMode mode) noexcept {
    fs_.open(s, std::ios_base::openmode(mode));
    PER_LOG_INFO << "open()";
    return fs_.is_open();
}

std::fstream::pos_type ReadAccessorImpl::readbinary(const hozon::netaos::core::Span<char> s) noexcept {
    fs_.read(s.data(), s.size());
    std::streamsize read_count = fs_.gcount();
    fs_.seekg(0, std::istream::beg);
    PER_LOG_INFO << "readbinary() " << read_count;
    return read_count;
}

std::fstream::pos_type ReadAccessorImpl::readtext(std::string& s) noexcept {
    std::ostringstream tmp;
    tmp << fs_.rdbuf();
    s = tmp.str();
    std::streamsize read_count = s.size();
    fs_.seekg(0, std::istream::beg);
    PER_LOG_INFO << "readtext() " << read_count;
    return read_count;
}

ReadAccessor& ReadAccessorImpl::readline(std::string& stream_string, char const delim /* = '\n' */) noexcept {
    std::vector<char> buf(40960);
    fs_.getline(buf.data(), buf.size(), delim);
    std::streamsize read_count = fs_.gcount();
    if (read_count > 0) {
        stream_string.assign(buf.data(), read_count);
    }
    PER_LOG_INFO << "readline() " << read_count;
    return *this;
}
}  // namespace per
}  // namespace netaos
}  // namespace hozon
