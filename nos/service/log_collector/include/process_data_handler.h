// 
// Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
// 
/// @file process_data_handler.h
/// @brief 
/// @author xumengjun@hozonauto.com
/// @version 1.0.0
/// @date 2023-11-15

#ifndef __LOG_COLLECTOR_INCLUDE_PROCESS_DATA_HANDLER_H__
#define __LOG_COLLECTOR_INCLUDE_PROCESS_DATA_HANDLER_H__

#include <vector>
#include <list>
#include <unordered_map>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include "log_collector/include/log_file_compression_base.h"
#include "log_collector/include/log_file_writer_base.h"
#include "log_collector/include/process_data.h"

namespace hozon {
namespace netaos {
namespace logcollector {

class ProcessDataHandler {
public:
    static ProcessDataHandler& Instance();

public:
    void Start();
    void Stop();
    void Wait();

    void AddFile(const LogFileWriterPtr &log_file_writer,
                const std::string &strappid, unsigned int process_id,
                unsigned int thread_id, off_t truncate_offset = 0);

    void AddHistoryFile(const std::string &strappid, const std::string &file_path,
                        const std::string &file_name, const std::string &file_suffix, int32_t file_seq);

    void CompressFile(const std::string &strappid, const std::string &file_path,
                        const std::string &file_name, std::string &zip_result_file);

private:
    ProcessDataHandler();
    ~ProcessDataHandler();

private:
    std::atomic<bool> running_{false};

    std::unordered_map<std::string, ProcessDataPtr> process_data_map_;
    std::mutex mutex_;

    std::thread flush_thread_;
    std::thread zip_thread_;

    std::shared_ptr<LogFileCompressionBase> log_file_compression_;
};

} // namespace logcollector
} // namespace netaos
} // namspace hozon

#endif // __LOG_COLLECTOR_INCLUDE_PROCESS_DATA_HANDLER_H__
