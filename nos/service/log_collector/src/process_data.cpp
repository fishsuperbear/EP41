// 
// Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
// 
/// @file process_data.cpp
/// @brief 
/// @author xumengjun@hozonauto.com
/// @version 1.0.0
/// @date 2023-11-15

#include "log_collector/include/process_data.h"

#include <cstring>

#include "log_collector/include/utils.h"
#include "log_collector/include/log_file_manager.h"
#include "log_collector/include/process_data_handler.h"

namespace hozon {
namespace netaos {
namespace logcollector {

ProcessData::ProcessData(ProcessDataHandler *process_data_handler, const std::string &appid, unsigned int process_id) :
    process_data_handler_(process_data_handler), appid_(appid),
    process_id_(process_id) {
    fileseq_filemask_map_.reserve(LogFileManager::max_file_seq_ << 1);
}

ProcessData::~ProcessData() {
}

void ProcessData::ExecFlush() {
    for (size_t i = 0; i < file_q_.size() && i < (size_t)LogFileManager::process_log_file_map_[appid_]->max_file_num;
                    ++i) {
        auto mask_info = file_q_[i];
        if (!mask_info->data_ready) {
            continue;
        }
        if (mask_info->flush_ready) {
            continue;
        }

        if (mask_info->thread_id == 0) {
            for (auto k = 0; k < mask_info->file_writer_num; ++k) {
                mask_info->file_writer_list[k]->CloseFile(mask_info->truncate_offset);
                mask_info->truncate_offset = 0;
            }
            mask_info->file_writer_list.clear();
            mask_info->flush_ready = true;
            fileseq_filemask_map_.erase(mask_info->file_seq);
        } else {
            for (auto k = 0; k < mask_info->file_writer_num; ++k) {
                mask_info->file_writer_list[k]->Flush();
            }
        }
    }
}

void ProcessData::ExecZip() {
    int index = 0;
    while (!file_q_.empty() && file_q_.rbegin() + index != file_q_.rend() - 1) {
        auto mask_info = *(file_q_.rbegin() + index);
        if (!mask_info->flush_ready) {
            break;
        }
        if (mask_info->zip_ready) {
            ++index;
            continue;
        }
        process_data_handler_->CompressFile(appid_, mask_info->file_path,
                                    mask_info->file_name, mask_info->zip_file_name);
        ++index;
        mask_info->zip_ready = true;
    }

    auto remove_file = [&](const std::string &file_name) {
        if (CommonTool::PathExists(file_name)) {
            if (std::remove(file_name.c_str()) == 0) {
                printf("remove file [%s] success\n", file_name.c_str());
            } else {
                printf("remove file [%s] failed. err[%s]\n",
                            file_name.c_str(), std::strerror(errno));
            }
        }
    };
    while (file_q_.size() > (size_t)LogFileManager::process_log_file_map_[appid_]->max_file_num) {
        auto &mask_info = file_q_.back();
        if (mask_info->data_ready && !mask_info->flush_ready) {
            for (auto k = 0; k < mask_info->file_writer_num; ++k) {
                mask_info->file_writer_list[k]->CloseFile(mask_info->truncate_offset);
                mask_info->truncate_offset = 0;
            }
            mask_info->file_writer_list.clear();
            mask_info->flush_ready = true;
            fileseq_filemask_map_.erase(mask_info->file_seq);
        }

        remove_file(mask_info->file_name);
        remove_file(mask_info->zip_file_name);

        file_q_.pop_back();
    }
}

void ProcessData::AddThreadFile(const LogFileWriterPtr &file_writer,
            unsigned int thread_id, off_t truncate_offset) {
    const std::string &file_name = file_writer->GetFileName();
    const std::string &file_path = file_writer->GetFilePath();
    int32_t file_seq = file_writer->GetFileSeq();
    FileMaskInfoPtr mask_info;
    if (fileseq_filemask_map_.count(file_seq) > 0) {
        mask_info = *fileseq_filemask_map_[file_seq];
        mask_info->thread_id ^= thread_id;
    } else {
        mask_info = std::make_shared<FileMaskInfo>();
        mask_info->file_writer_list.reserve(5000);
        mask_info->thread_id = thread_id;
        mask_info->data_ready = true;
        mask_info->file_name = file_name;
        mask_info->file_path = file_path;
        mask_info->file_seq = file_seq;

        file_q_.push_front(mask_info);
        fileseq_filemask_map_.insert(std::make_pair(file_seq, file_q_.begin()));

    }
    mask_info->truncate_offset = std::max(mask_info->truncate_offset, truncate_offset);
    mask_info->file_writer_list.push_back(file_writer);
    ++mask_info->file_writer_num;
}

void ProcessData::AddHistoryFile(const std::string &file_path, const std::string &file_name,
            const std::string &file_suffix, int32_t file_seq) {
    auto mask_info = std::make_shared<FileMaskInfo>(); 
    mask_info->file_name = file_name;
    mask_info->file_path = file_path;
    mask_info->file_seq = file_seq; 
    mask_info->data_ready = true;
    if (file_suffix == ".log") {
        mask_info->flush_ready = true;
    } else {
        mask_info->zip_ready = true;
    }

    file_q_.push_front(mask_info);
    fileseq_filemask_map_.insert(std::make_pair(file_seq, file_q_.begin()));
}

} // namespace logcollector
} // namespace netaos
} // namespace hozon
