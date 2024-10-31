// 
// Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
// 
/// @file config_manager.h
/// @brief 
/// @author xumengjun@hozonauto.com
/// @version 1.0.0
/// @date 2023-11-15

#ifndef __LOG_COLLECTOR_INCLUDE_CONFIG_MANAGER_H__
#define __LOG_COLLECTOR_INCLUDE_CONFIG_MANAGER_H__

#include <string>

#include "log_collector/include/log_file_writer_base.h"
#include "log_collector/include/log_file_compression_base.h"

namespace hozon {
namespace netaos {
namespace logcollector {

class ConfigManager {
public:
    static ConfigManager& Instance();

public:
    bool LoadConfig(const std::string &config_file);
    
    int32_t ConsumerThreadNum() const;

    const std::string& LogAppName() const;
    const std::string& LogAppDesc() const;
    const std::string& LogContextName() const;
    int32_t LogLevel() const;
    int32_t LogMode() const;
    const std::string& LogFilePath() const;
    int32_t MaxLogFileSize() const;
    int32_t MaxLogFileNum() const;

    CompressionMode LogCompressionMode() const;
    const std::string& CompressLogServiceName() const;

    LogFileWriterType LogWriterType() const;

    const std::string& GlobalLogConfigFile() const;

private:
    ConfigManager();
    ~ConfigManager();

private:
    int32_t consumer_thread_num_ = 1;
    std::string log_app_name_;
    std::string log_app_desc_;
    std::string log_app_context_name_; 
    int32_t log_level_ = 0;
    int32_t log_mode_ = 0;
    std::string log_file_path_ = "/opt/usr/log/soc_log/";
    int32_t max_log_file_size_ = 10 * 1024 * 1024;
    int32_t max_log_file_num_ = 10;

    CompressionMode log_compression_mode_ = CompressionMode::REMOTE_ZIPPER;
    std::string compression_log_service_name_;

    LogFileWriterType log_file_writer_type_ = LogFileWriterType::FWRITE;

    std::string global_log_config_file_;
};

} // namespace logcollector
} // namespace netaos
} // namespace hozon

#endif // __LOG_COLLECTOR_INCLUDE_CONFIG_MANAGER_H__
