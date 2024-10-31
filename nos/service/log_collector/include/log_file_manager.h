// 
// Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
// 
/// @file log_file_manager.h
/// @brief 
/// @author xumengjun@hozonauto.com
/// @version 1.0.0
/// @date 2023-11-15

#ifndef __LOG_COLLECTOR_INCLUDE_LOG_FILE_MANAGER_H__
#define __LOG_COLLECTOR_INCLUDE_LOG_FILE_MANAGER_H__

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>

#include "log_collector/include/log_file_writer_base.h"

namespace hozon {
namespace netaos {
namespace logcollector {

class LogFileManager {
public:
    static LogFileManager& Instance();
    
    LogFileWriterPtr GetLogFileWriter(const char* appid, unsigned int process_id, off_t needed_size) noexcept;

    static bool LoadGlobalLogConfig(const std::string &config_file);
    static void LoadHistoryLogFiles();

private:
    void CurrTimeStr(std::string &str_time);

private:
    LogFileManager();
    ~LogFileManager();

    LogFileManager(const LogFileManager &) = delete;
    LogFileManager& operator = (const LogFileManager &) = delete;

public:
    struct ProcessLogFile {
        std::string file_name;
        std::string file_path;
        int32_t max_file_num;
        off_t max_file_size;
        int32_t file_seq;
        std::atomic<off_t> write_offset;
        std::shared_ptr<std::mutex> mutex;
    };
    using ProcessLogFilePtr = std::shared_ptr<ProcessLogFile>;

    static std::unordered_map<std::string, LogFileManager::ProcessLogFilePtr> process_log_file_map_;
    static std::mutex log_file_writer_mutex_;
    static int32_t max_file_seq_; 

private:
    static thread_local LogFileManager instance_;
    
    std::string default_log_path_ = "/opt/usr/log/soc_log";
    off_t default_max_file_size_ = 1024 * 1024 * 10;
    int32_t default_max_file_num_ = 10;

    std::unordered_map<std::string, LogFileWriterPtr> process_filewriter_map_;
};

} // namespace logcollector
} // namespace netaos
} // namespace hozon

#endif // __LOG_COLLECTOR_INCLUDE_LOG_FILE_MANAGER_H__
