// 
// Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
// 
/// @file log_file_manager.cpp
/// @brief 
/// @author xumengjun@hozonauto.com
/// @version 1.0.0
/// @date 2023-11-15

#include "log_collector/include/log_file_manager.h"

#include <unistd.h>
#include <sys/time.h>
#include <sys/syscall.h>

#include <algorithm>
#include <fstream>
#include <unordered_set>

#include "json/json.h"

#include "log_collector/include/utils.h"
#include "log_collector/include/process_data_handler.h"
#include "log_collector/include/config_manager.h"
#include "log_collector/include/log_file_writer_factory.h"

#define NETA_DMB                              \
    do {                                      \
        __asm__ __volatile__("dmb st" ::      \
                                 : "memory"); \
    } while (0)

namespace hozon {
namespace netaos {
namespace logcollector {

int32_t LogFileManager::max_file_seq_ = 10000;
std::unordered_map<std::string, LogFileManager::ProcessLogFilePtr> LogFileManager::process_log_file_map_;
std::mutex LogFileManager::log_file_writer_mutex_;

static thread_local unsigned int LOCAL_THREAD_ID = syscall(SYS_gettid);
thread_local LogFileManager LogFileManager::instance_;

LogFileManager::LogFileManager() {
    process_filewriter_map_.reserve(1024);
}

LogFileManager::~LogFileManager() {
}

LogFileManager& LogFileManager::Instance() {
    return instance_; 
}

LogFileWriterPtr LogFileManager::GetLogFileWriter(const char *appid,
            unsigned int process_id, off_t needed_size) noexcept {
    std::string strappid(appid);
    if (process_filewriter_map_.count(strappid) != 0) {
        LogFileWriterPtr file_writer = process_filewriter_map_[strappid];
        ProcessLogFilePtr process_log_file = LogFileManager::process_log_file_map_[strappid];
        if (file_writer->GetFileSeq() == process_log_file->file_seq) {
            if (process_log_file->write_offset.load(std::memory_order_acquire) + needed_size <=
                        process_log_file->max_file_size) {
                auto begin_write_pos = process_log_file->write_offset.fetch_add(needed_size, std::memory_order_release);
                file_writer->FSeek(begin_write_pos);
                return file_writer;
            } else {
                ProcessLogFilePtr last_process_log_file = process_log_file;
                LogFileWriterPtr last_file_writer = file_writer;
                std::lock_guard<std::mutex> lock(*(last_process_log_file->mutex));

                process_log_file = LogFileManager::process_log_file_map_[strappid];
                bool create_file = false;
                if (last_process_log_file->file_seq == process_log_file->file_seq) {
                    create_file = true;
                    off_t offset = process_log_file->write_offset.load(std::memory_order_acquire);
                    ProcessDataHandler::Instance().AddFile(last_file_writer, appid,
                                                        process_id, LOCAL_THREAD_ID, offset);
                } else {
                    ProcessDataHandler::Instance().AddFile(last_file_writer, appid,
                                                        process_id, LOCAL_THREAD_ID);
                }

                auto new_file_writer = LogFileWriterFactory::Create(ConfigManager::Instance().LogWriterType());
                if (create_file) {
                    ProcessLogFilePtr new_process_log_file = std::make_shared<ProcessLogFile>();
                    new_process_log_file->max_file_size = process_log_file->max_file_size;
                    new_process_log_file->max_file_num = process_log_file->max_file_num;
                    new_process_log_file->file_path = std::move(process_log_file->file_path);
                    new_process_log_file->file_seq = process_log_file->file_seq;
                    new_process_log_file->mutex = process_log_file->mutex;
                    new_process_log_file->write_offset.store(0, std::memory_order_release);

                    std::string curr_time_str;
                    CurrTimeStr(curr_time_str);
                    char fname[512];
                    size_t name_len = snprintf(fname, sizeof(fname), "%s/%s_%04d_%s.log",
                                new_process_log_file->file_path.c_str(),
                                appid, ((++new_process_log_file->file_seq) % max_file_seq_),
                                curr_time_str.c_str());
                    new_process_log_file->file_name.assign(fname, name_len);
                    if (new_process_log_file->file_seq >= max_file_seq_) {
                        new_process_log_file->file_seq = new_process_log_file->file_seq % max_file_seq_;
                    }

                    new_file_writer->OpenFile(new_process_log_file->file_path, new_process_log_file->file_name,
                                                new_process_log_file->max_file_size,
                                                new_process_log_file->file_seq, true);
                    auto begin_write_pos = new_process_log_file->write_offset.fetch_add(needed_size,
                                                std::memory_order_release);
                    new_file_writer->FSeek(begin_write_pos);

                    LogFileManager::process_log_file_map_[strappid] = new_process_log_file;
                    process_filewriter_map_[strappid] = new_file_writer;
                } else {
                    new_file_writer->OpenFile(process_log_file->file_path, process_log_file->file_name,
                                                process_log_file->max_file_size, process_log_file->file_seq);
                    auto begin_write_pos = process_log_file->write_offset.fetch_add(needed_size, 
                                std::memory_order_release);
                    new_file_writer->FSeek(begin_write_pos);
                    process_filewriter_map_[strappid] = new_file_writer;
                }
                
                ProcessDataHandler::Instance().AddFile(new_file_writer, appid, process_id, LOCAL_THREAD_ID);
                return new_file_writer;
            }
        } else {
            ProcessDataHandler::Instance().AddFile(file_writer, appid, process_id, LOCAL_THREAD_ID);

            auto new_file_writer = LogFileWriterFactory::Create(ConfigManager::Instance().LogWriterType());
            new_file_writer->OpenFile(process_log_file->file_path, process_log_file->file_name,
                                        process_log_file->max_file_size, process_log_file->file_seq);
            auto begin_write_pos = process_log_file->write_offset.fetch_add(needed_size, std::memory_order_release);
            new_file_writer->FSeek(begin_write_pos);
            process_filewriter_map_[strappid] = new_file_writer;
            ProcessDataHandler::Instance().AddFile(new_file_writer, appid, process_id, LOCAL_THREAD_ID);
            return new_file_writer;
        }
    } else {
        std::lock_guard<std::mutex> lock(LogFileManager::log_file_writer_mutex_);
        bool create_file = false;
        if (LogFileManager::process_log_file_map_.count(strappid) == 0) {
            create_file = true;

            ProcessLogFilePtr process_log_file = std::make_shared<ProcessLogFile>();
            process_log_file->file_seq = -1;
            process_log_file->mutex = std::make_shared<std::mutex>();
            process_log_file->write_offset.store(0, std::memory_order_release);
            process_log_file->max_file_size = default_max_file_size_;
            process_log_file->max_file_num = default_max_file_num_;
            process_log_file->file_path = default_log_path_;

            std::string curr_time_str;
            CurrTimeStr(curr_time_str);
            char fname[512];
            size_t name_len = snprintf(fname, sizeof(fname), "%s/%s_%04d_%s.log",
                        default_log_path_.c_str(),
                        appid, (++process_log_file->file_seq) % max_file_seq_,
                        curr_time_str.c_str());
            process_log_file->file_name.assign(fname, name_len);
            if (process_log_file->file_seq >= max_file_seq_) {
                process_log_file->file_seq = process_log_file->file_seq % max_file_seq_;
            }

            LogFileManager::process_log_file_map_[strappid] = process_log_file;
        } else {
            ProcessLogFilePtr process_log_file = LogFileManager::process_log_file_map_[strappid];
            if (process_log_file->file_name.empty()) {
                std::string curr_time_str;
                CurrTimeStr(curr_time_str);
                char fname[512];
                size_t name_len = snprintf(fname, sizeof(fname), "%s/%s_%04d_%s.log",
                            process_log_file->file_path.c_str(),
                            appid, (++process_log_file->file_seq) % max_file_seq_,
                            curr_time_str.c_str());
                process_log_file->file_name.assign(fname, name_len);
                if (process_log_file->file_seq >= max_file_seq_) {
                    process_log_file->file_seq = process_log_file->file_seq % max_file_seq_;
                }

                create_file = true;
            }
        }
        
        ProcessLogFilePtr plf = LogFileManager::process_log_file_map_[strappid]; 
        auto file_writer = LogFileWriterFactory::Create(ConfigManager::Instance().LogWriterType());
        file_writer->OpenFile(plf->file_path, plf->file_name, plf->max_file_size, plf->file_seq, create_file);
        auto begin_write_pos = plf->write_offset.fetch_add(needed_size, std::memory_order_release);
        file_writer->FSeek(begin_write_pos);
        process_filewriter_map_[strappid] = file_writer;
        
        ProcessDataHandler::Instance().AddFile(file_writer, appid, process_id, LOCAL_THREAD_ID);

        return file_writer;
    }
}

void LogFileManager::CurrTimeStr(std::string &str_time) {
    time_t t = time(nullptr);
    tm buf;
    char mbstr[64];
    if (localtime_r(&t, &buf) != nullptr) {
        size_t n = strftime(mbstr, sizeof(mbstr), "%Y-%m-%d_%H-%M-%S", &buf);
        if (n > 0) {
            str_time.assign(mbstr, n);
        }
    } else {
        str_time = "1900-01-01_00-00-00";
    }
}

bool LogFileManager::LoadGlobalLogConfig(const std::string &config_file) {
    std::ifstream infile(config_file, std::ios::binary);
    if (!infile.is_open()) {
        printf("open file failed. %s:\n", config_file.c_str());
        return false;
    }

    Json::CharReaderBuilder read_builder;
    Json::Value root;
    std::string err;

    if (!Json::parseFromStream(read_builder, infile, &root, &err)) {
        printf("can't parser config file, %s\n", config_file.c_str());
        infile.close();
        return false;
    }

    if (root["log_config"].isNull()) {
        infile.close();
        return false;
    }

    int config_size = root["log_config"].size();
    auto &config_value = root["log_config"];
    for (int i = 0; i < config_size; ++i) {
        const std::string &log_app_name = config_value[i]["LogAppName"].asString();
        const std::string &log_file_path = config_value[i]["LogFilePath"].asString();
        int32_t max_file_num = config_value[i]["MaxLogFileNum"].asInt();
        int32_t max_log_file_size = config_value[i]["MaxSizeOfLogFile"].asInt();
        if (LogFileManager::process_log_file_map_.count(log_app_name) > 0) {
            printf("log app name repeated!!!, pass");
            continue;
        }

        ProcessLogFilePtr process_log_file = std::make_shared<ProcessLogFile>();
        process_log_file->file_path = log_file_path;
        process_log_file->max_file_num = max_file_num;
        process_log_file->max_file_size = max_log_file_size * 1024 * 1024;
        process_log_file->file_seq = -1;
        process_log_file->mutex = std::make_shared<std::mutex>();
        process_log_file->write_offset.store(0, std::memory_order_release);
        LogFileManager::process_log_file_map_.insert(std::make_pair(log_app_name, process_log_file));
    }

    infile.close();
    return true;
}

void LogFileManager::LoadHistoryLogFiles() {
    std::unordered_set<std::string> pathset;
    for (auto &[_, process_log] : LogFileManager::process_log_file_map_) {
        pathset.insert(process_log->file_path);
    }

    struct FileDesc {
        std::string appid;
        std::string file_name;
        std::string file_path;
        std::string file_suffix;
        int32_t file_seq;
        uint64_t create_time;
    };
    using FileDescPtr = std::shared_ptr<FileDesc>;

    std::vector<std::string> elems;
    std::vector<FileDescPtr> file_desc_array;
    for (auto &log_path : pathset) {
        std::vector<std::string> history_files;
        CommonTool::ListSubPaths(log_path, DT_REG, "", history_files);
        for (auto &iter : history_files) {
            if (iter == "." || iter == ".." || iter[0] == '.') {
                continue;
            }
            std::string file_base_name = iter;
            auto pos = iter.rfind("/");
            if (pos != std::string::npos) {
                file_base_name = iter.substr(pos + 1, iter.size());
            }
            const std::string &file_suffix = iter.substr(iter.size() - 4, 4);
            if (file_suffix != ".log" && file_suffix != ".zip" && file_suffix != ".zst") {
                continue;
            }
            elems.clear(); 
            CommonTool::SplitStr(file_base_name, '_', elems);
            if (elems.size() < 4) {
                continue;
            }
            int32_t file_seq = std::stoi(elems[elems.size() - 3]);
            std::string appid;
            for (auto i = 0; i < (int)elems.size() - 3; ++i) {
                if (appid.empty()) {
                    appid = elems[i];
                } else {
                    appid += "_" + elems[i];
                }
            }
            if (LogFileManager::process_log_file_map_.count(appid) == 0) {
                continue;
            }
            auto &process_log_file = LogFileManager::process_log_file_map_[appid];
            process_log_file->file_seq = std::max(process_log_file->file_seq, file_seq);
            auto file_desc = std::make_shared<FileDesc>();
            file_desc->appid = appid;
            auto time_pair = CommonTool::GetFileCreateTime(iter.c_str());
            file_desc->create_time = time_pair.first * 1000000000ull + time_pair.second;
            file_desc->file_name = iter;
            file_desc->file_path = log_path;
            file_desc->file_suffix = file_suffix;
            file_desc->file_seq = file_seq;
            file_desc_array.push_back(file_desc);
        }
    }
    std::stable_sort(file_desc_array.begin(), file_desc_array.end(),
                [&](const FileDescPtr &f1, const FileDescPtr &f2) { return f1->create_time < f2->create_time; });

    for (auto &file_desc : file_desc_array) {
        if (file_desc->file_suffix == ".zip" || file_desc->file_suffix == ".zst") {
            ProcessDataHandler::Instance().AddHistoryFile(file_desc->appid, file_desc->file_path,
                        file_desc->file_name, file_desc->file_suffix, file_desc->file_seq);
        }
    }

    for (auto &file_desc : file_desc_array) {
        if (file_desc->file_suffix == ".log") {
            ProcessDataHandler::Instance().AddHistoryFile(file_desc->appid, file_desc->file_path,
                        file_desc->file_name, file_desc->file_suffix, file_desc->file_seq);
        }
    }
}

} // namespace logcollector
} // namespace netaos
} // namespace hozon
