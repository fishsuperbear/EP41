// 
// Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
// 
/// @file process_data.h
/// @brief 
/// @author xumengjun@hozonauto.com
/// @version 1.0.0
/// @date 2023-11-15

#ifndef __LOG_COLLECTOR_INCLUDE_PROCESS_DATA_H__
#define __LOG_COLLECTOR_INCLUDE_PROCESS_DATA_H__

#include <unordered_map>
#include <queue>
#include <string>
#include <vector>
#include <memory>
#include <atomic>

#include "log_collector/include/thread_safe_queue.h" 
#include "log_collector/include/log_file_writer_base.h"

namespace hozon {
namespace netaos {
namespace logcollector {

class ProcessData;
using ProcessDataPtr = std::shared_ptr<ProcessData>;

class ProcessDataHandler;

class ProcessData {
public:
    explicit ProcessData(ProcessDataHandler *process_data_handler, const std::string &appid,
                unsigned int process_id);
    ~ProcessData();

public:
    void ExecFlush();
    void ExecZip();

    void AddThreadFile(const LogFileWriterPtr &file_writer, unsigned int thread_id,
                off_t truncate_offset);
    void AddHistoryFile(const std::string &file_path, const std::string &file_name,
                const std::string &file_suffix, int32_t file_seq); 

protected:
    ProcessDataHandler *process_data_handler_;
    std::string appid_;
    unsigned int process_id_;

    struct FileMaskInfo {
        std::string file_name;
        std::string file_path;
        std::vector<LogFileWriterPtr> file_writer_list;
        int file_writer_num;
        off_t truncate_offset;
        unsigned int thread_id;
        int32_t file_seq;
        std::string zip_file_name;
        std::atomic<bool> data_ready;
        std::atomic<bool> flush_ready;
        std::atomic<bool> zip_ready;

        FileMaskInfo() : file_writer_num(0), truncate_offset(0),
                    thread_id(0), file_seq(0), data_ready{false},
                    flush_ready{false}, zip_ready{false} {
        }

        void Reset() {
            file_writer_list.clear();
            file_writer_num = 0;
            truncate_offset = 0;
            thread_id = 0;
            file_seq = 0;
            data_ready.store(false);
            flush_ready.store(false);
            zip_ready.store(false);
        }
    };
    using FileMaskInfoPtr = std::shared_ptr<FileMaskInfo>;

    std::deque<FileMaskInfoPtr> file_q_;
    std::unordered_map<int32_t, std::deque<FileMaskInfoPtr>::iterator> fileseq_filemask_map_;
};

} // namespace logcollector
} // namespace netaos
} // namespace hozon

#endif // __LOG_COLLECTOR_INCLUDE_PROCESS_DATA_H__
