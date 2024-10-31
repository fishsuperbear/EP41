// 
// Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
// 
/// @file log_file_memcpy_writer.h
/// @brief 
/// @author xumengjun@hozonauto.com
/// @version 1.0.0
/// @date 2023-11-15

#ifndef __LOG_COLLECTOR_INCLUDE_LOG_FILE_MEMCPY_WRITER_H__
#define __LOG_COLLECTOR_INCLUDE_LOG_FILE_MEMCPY_WRITER_H__

#include <string>
#include <memory>

#include "log_collector/include/log_file_writer_base.h"

namespace hozon {
namespace netaos {
namespace logcollector {

class LogFileMemcpyWriter;
using LogFileMemcpyWriterPtr = std::shared_ptr<LogFileMemcpyWriter>;

class LogFileMemcpyWriter : public LogFileWriterBase {
public:
    LogFileMemcpyWriter();
    ~LogFileMemcpyWriter();

public:
    bool OpenFile(const std::string &file_path, const std::string &filename,
                off_t max_file_size, uint32_t file_seq, bool create_file = false) override;
    void CloseFile(off_t truncate_offset) override;

    bool AddData(const iovec *iov, size_t count, size_t len) noexcept override;
    bool AddData(const char *data, size_t len) noexcept override;

    void FSeek(off_t offset) override;
    void Flush() override;

    const std::string& GetFilePath() override;
    const std::string& GetFileName() override;
    int32_t GetFileSeq() override;

protected:
    void Ftruncate(off_t offset); 
    
protected:
    std::string file_path_;
    std::string file_name_;
    int32_t file_seq_ = 0;
    int fd_ = -1;
    char *file_addr_ = nullptr;
    off_t curr_write_pos_ = 0;
    off_t max_file_size_ = 0;
};

} // namespace logcollector
} // namespace netaos
} // namespace hozon

#endif // __LOG_COLLECTOR_INCLUDE_LOG_FILE_MEMCPY_WRITER_H__
