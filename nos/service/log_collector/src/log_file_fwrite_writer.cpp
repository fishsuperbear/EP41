// 
// Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
// 
/// @file log_file_fwrite_writer.cpp
/// @brief 
/// @author xumengjun@hozonauto.com
/// @version 1.0.0
/// @date 2023-11-15

#include "log_collector/include/log_file_fwrite_writer.h"

#include <unistd.h>
#include <sys/time.h>
#include <sys/syscall.h>
#include <fcntl.h>

#include <sstream>
#include <algorithm>
#include <thread>
#include <cstring>

#include "log_collector/include/utils.h"

namespace hozon {
namespace netaos {
namespace logcollector {

LogFileFwriteWriter::LogFileFwriteWriter() : LogFileWriterBase() {
}

LogFileFwriteWriter::~LogFileFwriteWriter() {
    this->CloseFile(1);
}

bool LogFileFwriteWriter::OpenFile(const std::string &file_path, const std::string &filename,
                off_t max_file_size, uint32_t file_seq, bool create_file) {
    file_path_ = file_path; 
    file_name_ = filename;
    max_file_size_ = max_file_size;
    file_seq_ = file_seq;

    if (create_file) {
        FILE *fp = std::fopen(file_name_.c_str(), "wb+");
        std::fclose(fp);
    }

    fd_ = std::fopen(file_name_.c_str(), "r+");
    if (!fd_) {
        printf("open file failed.[%s][%s]\n", filename.c_str(), std::strerror(errno));
        return false;
    }
    return true;
}

void LogFileFwriteWriter::FSeek(off_t offset) {
    fseeko(fd_, offset, SEEK_SET);
}

bool LogFileFwriteWriter::AddData(const iovec *iov, size_t count, size_t len) {
    size_t write_len = 0;
    for (size_t i = 0; i < count; ++i) {
        write_len += std::fwrite((char*)iov[i].iov_base, 1, iov[i].iov_len, fd_);
    }

    return write_len == len;
}

bool LogFileFwriteWriter::AddData(const char *data, size_t len) {
    size_t write_len = std::fwrite(data, 1, len, fd_);
    return write_len == len;
}

void LogFileFwriteWriter::Flush() {
    if (fd_)
      std::fflush(fd_);
}

const std::string& LogFileFwriteWriter::GetFilePath() {
    return file_path_;
}

const std::string& LogFileFwriteWriter::GetFileName() {
    return file_name_;
}

int32_t LogFileFwriteWriter::GetFileSeq() {
    return file_seq_;
}

void LogFileFwriteWriter::CloseFile(off_t truncate_offset) {
    if (!fd_) {
        return;
    }

    if (fsync(fileno(fd_)) != 0) {
        printf("fsync file failed.[%s][%s]\n", file_name_.c_str(), std::strerror(errno));
    }

    if (posix_fadvise(fileno(fd_), 0, 0, POSIX_FADV_DONTNEED) != 0) {
        printf("fadvise file failed.[%s][%s]\n", file_name_.c_str(), std::strerror(errno));
    }

    std::fclose(fd_);
    fd_ = nullptr;
}

} // namespace logcollector 
} // namespace netaos
} // namespace hozon
