// 
// Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
// 
/// @file log_file_writer_base.h
/// @brief 
/// @author xumengjun@hozonauto.com
/// @version 1.0.0
/// @date 2023-11-15

#ifndef __LOG_COLLECTOR_INCLUDE_LOG_FILE_WRITER_BASE_H__
#define __LOG_COLLECTOR_INCLUDE_LOG_FILE_WRITER_BASE_H__

#include <string>
#include <memory>
#include <sys/uio.h>

namespace hozon {
namespace netaos {
namespace logcollector {

class LogFileWriterBase;
using LogFileWriterPtr = std::shared_ptr<LogFileWriterBase>;

enum class LogFileWriterType : int32_t {
    NONE = -1,
    FWRITE = 0,
    MEMCPY
};

class LogFileWriterBase {
public:
    LogFileWriterBase() {}
    virtual ~LogFileWriterBase() {}

public:
    virtual bool OpenFile(const std::string &file_path, const std::string &filename,
                    off_t max_file_size, uint32_t file_seq, bool create_file = false) = 0;
    virtual void CloseFile(off_t truncate_offset) = 0;

    virtual bool AddData(const iovec *iov, size_t count, size_t len) = 0;
    virtual bool AddData(const char *data, size_t len) = 0;

    virtual void FSeek(off_t offset) = 0;
    virtual void Flush() = 0;

    virtual const std::string& GetFilePath() = 0;
    virtual const std::string& GetFileName() = 0;
    virtual int32_t GetFileSeq() = 0;
};

} // namespace logcollector
} // namespace netaos
} // namespace hozon

#endif // __LOG_COLLECTOR_INCLUDE_LOG_FILE_WRITER_BASE_H__
