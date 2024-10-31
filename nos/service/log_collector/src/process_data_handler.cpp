// 
// Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
// 
/// @file process_data_handler.cpp
/// @brief 
/// @author xumengjun@hozonauto.com
/// @version 1.0.0
/// @date 2023-11-15

#include "log_collector/include/process_data_handler.h"

#include <algorithm>
#include <queue>
#include <memory>
#include <vector>
#include <cstring>

#include "log_collector/include/utils.h"
#include "log_collector/include/log_file_manager.h"
#include "log_collector/include/config_manager.h"
#include "log_collector/include/log_file_local_zipper_compression.h"
#include "log_collector/include/log_file_remote_zipper_compression.h"

#define NETA_DMB                              \
    do {                                      \
        __asm__ __volatile__("dmb st" ::      \
                                 : "memory"); \
    } while (0)


namespace hozon {
namespace netaos {
namespace logcollector {

ProcessDataHandler::ProcessDataHandler() {
}

ProcessDataHandler& ProcessDataHandler::Instance() {
    static ProcessDataHandler thread_data_manager;
    return thread_data_manager;
}

ProcessDataHandler::~ProcessDataHandler() {
}

void ProcessDataHandler::Start() {
    auto flush_action = [this] {
        while (running_.load()) {
            for (auto &[_, process_data] : process_data_map_) {
                process_data->ExecFlush();
            } 
            //std::this_thread::sleep_for(std::chrono::milliseconds(200));
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    };

    auto zip_action = [this] {
        while (running_.load()) {
            for (auto &[_, process_data] : process_data_map_) {
                process_data->ExecZip();
            } 
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    };

    running_.store(true);
    flush_thread_ = std::thread(flush_action);
    zip_thread_ = std::thread(zip_action);

    if (ConfigManager::Instance().LogCompressionMode() == CompressionMode::LOCAL_ZIPPER) {
        log_file_compression_ = std::make_shared<LogFileLocalZipperCompression>();
    } else if (ConfigManager::Instance().LogCompressionMode() == CompressionMode::REMOTE_ZIPPER) {
        auto &log_service_name = ConfigManager::Instance().CompressLogServiceName();
        if (log_service_name.empty()) {
            log_file_compression_ = std::make_shared<LogFileRemoteZipperCompression>();
        } else {
            log_file_compression_ = std::make_shared<LogFileRemoteZipperCompression>(log_service_name);
        }
    } else {
        printf("unknown compression mode\n");
    }
}

void ProcessDataHandler::Stop() {
    running_.store(false);
}

void ProcessDataHandler::Wait() {
    flush_thread_.join();
    zip_thread_.join();
}

void ProcessDataHandler::AddFile(const LogFileWriterPtr &log_file_writer,
            const std::string &strappid, unsigned int process_id,
            unsigned int thread_id, off_t truncate_offset) {
    if (process_data_map_.count(strappid) == 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (process_data_map_.count(strappid) == 0) {
            process_data_map_[strappid] = std::make_shared<ProcessData>(this, strappid, process_id);
        }
        NETA_DMB;
    }
    process_data_map_[strappid]->AddThreadFile(log_file_writer, thread_id, truncate_offset);
}

void ProcessDataHandler::AddHistoryFile(const std::string &strappid, const std::string &file_path,
            const std::string &file_name, const std::string &file_suffix, int32_t file_seq) {
    if (process_data_map_.count(strappid) == 0) {
        process_data_map_[strappid] = std::make_shared<ProcessData>(this, strappid, -1);
    }
    process_data_map_[strappid]->AddHistoryFile(file_path, file_name, file_suffix, file_seq);
}

void ProcessDataHandler::CompressFile(const std::string &strappid, const std::string &file_path,
                    const std::string &file_name, std::string &zip_result_file) {
    if (log_file_compression_) {
        log_file_compression_->DO(strappid, file_path, file_name, zip_result_file);
    }
}

} // namespace logcollector
} // namespace netaos
} // namespace hozon
