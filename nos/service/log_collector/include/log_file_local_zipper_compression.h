// 
// Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
// 
/// @file log_file_local_zipper_compression.h
/// @brief 
/// @author xumengjun@hozonauto.com
/// @version 1.0.0
/// @date 2023-11-15

#ifndef __LOG_COLLECTOR_INCLUDE_LOG_FILE_LOCAL_COMPRESSION_H__
#define __LOG_COLLECTOR_INCLUDE_LOG_FILE_LOCAL_COMPRESSION_H__

#include <string>

#include "log_collector/include/log_file_compression_base.h"

namespace hozon {
namespace netaos {
namespace logcollector {

class LogFileLocalZipperCompression : public LogFileCompressionBase {
public:
    LogFileLocalZipperCompression();
    ~LogFileLocalZipperCompression();

public:
    bool DO(const std::string &appid, const std::string &file_path,
                const std::string &file_name, std::string &zip_result_file) override;
};

}
}
}
#endif // __LOG_COLLECTOR_INCLUDE_LOG_FILE_LOCAL_COMPRESSION_H__
