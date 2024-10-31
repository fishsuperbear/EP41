// 
// Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
// 
/// @file log_file_memcpy_writer.cpp
/// @brief 
/// @author xumengjun@hozonauto.com
/// @version 1.0.0
/// @date 2023-11-15

#include "log_collector/include/log_file_memcpy_writer.h"

#include <unistd.h>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>
#include <sys/syscall.h>
#include <sys/mman.h>

#include <sstream>
#include <algorithm>
#include <thread>

#include "log_collector/include/utils.h"

namespace hozon {
namespace netaos {
namespace logcollector {

LogFileMemcpyWriter::LogFileMemcpyWriter() : LogFileWriterBase() {
}

LogFileMemcpyWriter::~LogFileMemcpyWriter() {
    CloseFile(0);
}

bool LogFileMemcpyWriter::OpenFile(const std::string &file_path, const std::string &filename,
            off_t max_file_size, uint32_t file_seq, bool create_file) {
    file_name_ = filename;
    file_path_ = file_path; 
    file_seq_ = file_seq;
    max_file_size_ = max_file_size;

    int flags = O_RDWR;
    if (create_file) {
       flags |= O_CREAT;
    }
    fd_ = open(file_name_.c_str(), flags, S_IRUSR | S_IWUSR);
    if (fd_ == -1) {
        printf("open file failed.[%s][%s]\n", filename.c_str(), std::strerror(errno));
        return false;
    }
    
    if (create_file) {
        if (ftruncate(fd_, max_file_size_) == -1) {
            printf("ftruncate file failed.[%s][%s]\n", filename.c_str(), std::strerror(errno));
            return false;
        }
    }

    file_addr_ = static_cast<char*>(mmap(nullptr, max_file_size_, PROT_WRITE, MAP_SHARED, fd_, 0));
    if (file_addr_ == MAP_FAILED) {
        printf("mmap file failed.[%s][%s]\n", file_name_.c_str(), std::strerror(errno));
        return false;
    }

    return true;
}

const std::string& LogFileMemcpyWriter::GetFilePath() {
    return file_path_;
}

const std::string& LogFileMemcpyWriter::GetFileName() {
    return file_name_;
}

int32_t LogFileMemcpyWriter::GetFileSeq() {
    return file_seq_;
}

void LogFileMemcpyWriter::FSeek(off_t offset) {
    curr_write_pos_ = offset;
}

void LogFileMemcpyWriter::Ftruncate(off_t offset) {
    if (offset > 0 && ftruncate(fd_, offset) != 0) {
        printf("ftruncate file failed.[%s][%s]\n", file_name_.c_str(), std::strerror(errno));
    }

    if (fsync(fd_) != 0) {
        printf("fsync file failed.[%s][%s]\n", file_name_.c_str(), std::strerror(errno));
    }

    if (posix_fadvise(fd_, 0, 0, POSIX_FADV_DONTNEED) != 0) {
        printf("fadvise file failed.[%s][%s]\n", file_name_.c_str(), std::strerror(errno));
    }
}

bool LogFileMemcpyWriter::AddData(const iovec *iov, size_t count, size_t len) noexcept {
    for (size_t i = 0; i < count; ++i) {
        std::memcpy(file_addr_ + curr_write_pos_, (char*)iov[i].iov_base, iov[i].iov_len);
        curr_write_pos_ += iov[i].iov_len;
    }
    return true;
}

bool LogFileMemcpyWriter::AddData(const char *data, size_t len) noexcept {
    std::memcpy(file_addr_, data, len);
    return true;
}

void LogFileMemcpyWriter::Flush() {
    if (file_addr_) {
        msync(file_addr_, max_file_size_, MS_ASYNC);
    }
}

void LogFileMemcpyWriter::CloseFile(off_t truncate_offset) {
    if (fd_ == -1) {
        return;
    }

    if (file_addr_) {
        if (munmap(file_addr_, max_file_size_) == -1) {
            printf("munmap file failed.[%s][%s]\n", file_name_.c_str(), std::strerror(errno));
        }
        file_addr_ = nullptr;
    }

    Ftruncate(truncate_offset);

    close(fd_);
    fd_ = -1;
}

} // namespace logcollector 
} // namespace netaos
} // namespace hozon
