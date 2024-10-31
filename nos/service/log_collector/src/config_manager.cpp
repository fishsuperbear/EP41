// 
// Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
// 
/// @file config_manager.cpp
/// @brief 
/// @author xumengjun@hozonauto.com
/// @version 1.0.0
/// @date 2023-11-15

#include "log_collector/include/config_manager.h"

#include <unistd.h>
#include <cstring>
#include <memory>
#include <fstream>

#include "json/json.h"

namespace hozon {
namespace netaos {
namespace logcollector {

ConfigManager::ConfigManager() {
}

ConfigManager& ConfigManager::Instance() {
    static ConfigManager config_manager;
    return config_manager;
}

ConfigManager::~ConfigManager() {
}

bool ConfigManager::LoadConfig(const std::string &config_file) {
    if (access(config_file.c_str(), F_OK) != 0) {
        printf("can't find config file: %s, err: %s, exit\n", config_file.c_str(), std::strerror(errno));
        return false;
    }

    Json::Value root_reader;
    Json::CharReaderBuilder read_builder;
    std::ifstream ifs(config_file);
    std::unique_ptr<Json::CharReader> reader(read_builder.newCharReader());
    JSONCPP_STRING errs;
    if (!Json::parseFromStream(read_builder, ifs, &root_reader, &errs)) {
        printf("can't parse config file %s, err: %s, exit\n", config_file.c_str(), std::strerror(errno));
        return false;
    }
    consumer_thread_num_ = root_reader["consumer_thread_num"].asInt();
    log_app_name_ = root_reader["LogAppName"].asString(); 
    log_app_desc_ = root_reader["LogAppDescription"].asString();
    log_app_context_name_ = root_reader["LogContextName"].asString();
    log_level_ = root_reader["LogLevel"].asUInt();
    log_mode_ = root_reader["LogMode"].asUInt();
    if (root_reader.isMember("LogCompressionMode")) {
        auto compression_mode = root_reader["LogCompressionMode"].asInt();
        if (compression_mode == 1) {
            log_compression_mode_ = CompressionMode::REMOTE_ZIPPER;
        } else if (compression_mode == 2) {
            log_compression_mode_ = CompressionMode::LOCAL_ZIPPER;
        } else {
            log_compression_mode_ = CompressionMode::NONE;
        }
    }
    if (root_reader.isMember("LogFileWriterType")) {
        auto writer_type = root_reader["LogFileWriterType"].asInt();
        if (writer_type == 1) {
            log_file_writer_type_ = LogFileWriterType::FWRITE;
        } else {
            log_file_writer_type_ = LogFileWriterType::MEMCPY;
        }
    }

    global_log_config_file_ = root_reader["GlobalLogConfigFile"].asString();

    return true;
}

int32_t ConfigManager::ConsumerThreadNum() const {
    return consumer_thread_num_;
}

const std::string& ConfigManager::LogAppName() const {
    return log_app_name_;
}

const std::string& ConfigManager::LogContextName() const {
    return log_app_context_name_;
}

const std::string& ConfigManager::LogAppDesc() const {
    return log_app_desc_;
}

int32_t ConfigManager::LogLevel() const {
    return log_level_;
}

int32_t ConfigManager::LogMode() const {
    return log_mode_;
}

const std::string& ConfigManager::LogFilePath() const {
    return log_file_path_;
}

int32_t ConfigManager::MaxLogFileSize() const {
    return max_log_file_size_;
}

int32_t ConfigManager::MaxLogFileNum() const {
    return max_log_file_num_;
}

CompressionMode ConfigManager::LogCompressionMode() const {
    return log_compression_mode_;
}

const std::string& ConfigManager::CompressLogServiceName() const {
    return compression_log_service_name_;
}

LogFileWriterType ConfigManager::LogWriterType() const {
    return log_file_writer_type_;
}

const std::string& ConfigManager::GlobalLogConfigFile() const {
    return global_log_config_file_;
}

} // namespace logcollector
} // namespace netaos
} // namspace hozon
