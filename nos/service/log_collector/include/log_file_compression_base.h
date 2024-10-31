// 
// Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
// 
/// @file log_file_compression_base.h
/// @brief 
/// @author xumengjun@hozonauto.com
/// @version 1.0.0
/// @date 2023-11-15

#ifndef __LOG_COLLECTOR_INCLUDE_LOG_FILE_COMPRESSION_BASE_H__
#define __LOG_COLLECTOR_INCLUDE_LOG_FILE_COMPRESSION_BASE_H__

#include <string>
#include <cstdint>

#include "spdlog/common.h"
#include "spdlog/details/file_helper.h"

namespace hozon {
namespace netaos {
namespace logcollector {

enum class CompressionMode : int32_t {
    NONE = -1,
    LOCAL_ZIPPER = 0,
    REMOTE_ZIPPER
};

class LogFileCompressionBase {
public:
    LogFileCompressionBase();
    virtual ~LogFileCompressionBase();

public:
    virtual bool DO(const std::string &appid, const std::string &file_path, const std::string &file_name,
                        std::string &zip_result_file) = 0;

protected:
    bool RenameFile(const spdlog::filename_t &src_filename, const spdlog::filename_t &target_filename);
    std::string GetAbsolutePath(const std::string& relative_path);
};

} // namespace logcollector
} // namespace netaos
} // namspace hozon
#endif // __LOG_COLLECTOR_INCLUDE_LOG_FILE_COMPRESSION_BASE_H__
