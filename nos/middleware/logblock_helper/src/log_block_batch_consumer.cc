// 
// Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
// 
/// @file log_block_batch_consumer.cc
/// @brief 
/// @author xumengjun@hozonauto.com
/// @version 1.0.0
/// @date 2023-11-07

#include "logblock_helper/include/log_block_batch_consumer.h"

#include <cstdlib>
#include <cstring>
#include <ctime>
#include <sstream>

namespace hozon {
namespace netaos {
namespace logblock {

LogBlockBatchConsumer::LogBlockBatchConsumer() {
    srand(time(NULL));
}

LogBlockBatchConsumer::~LogBlockBatchConsumer() {
}

LogBlockBatchConsumer& LogBlockBatchConsumer::Instance() {
    static LogBlockBatchConsumer log_block_consumer;
    return log_block_consumer;
}

bool LogBlockBatchConsumer::Start(int thread_num, const CallBackFunc &data_callback_func) {
    if (inited_) {
        return false;
    }

    thread_num_ = thread_num;
    data_callback_func_ = data_callback_func;

    if (thread_num_ <= 0) {
        return false;
    }
    for (int i = 0; i < thread_num_; ++i) {
        threads_.emplace_back([this]{
            this->Run();
        });
    }
    inited_ = true;

    return true;
}

void LogBlockBatchConsumer::Run() {
    LogBlockReaderThreadHandle log_block_reader_thread_handle;
    if (!OpenLogBlockThreadHandle(log_block_reader_thread_handle)) {
        return;
    }

    LogBlockReaderInfo log_block_reader_info;
    LOGBLOCK_READRETSTATUS read_status;
    while (running_.load()) {
        auto ret = GetNextLogBlockToRead(log_block_reader_thread_handle, &log_block_reader_info, 100, &read_status);
        if (ret == 0 && read_status == LOGBLOCK_READRETSTATUS_GET) {
            HandleData(log_block_reader_info);
        }
    }

    FinishLogBlockRead(log_block_reader_thread_handle);
}

void LogBlockBatchConsumer::HandleData(LogBlockReaderInfo &log_block_reader_info) {
    auto process_id = log_block_reader_info.producerthreadinfo.process_id;
    auto thread_id = log_block_reader_info.producerthreadinfo.thread_id;
    const char *name = log_block_reader_info.producerthreadinfo.name;

    u32 begin = log_block_reader_info.roffset_begin;
    u32 end = log_block_reader_info.roffset_end;

    u32 read_size = log_block_reader_info.roffset_end - log_block_reader_info.roffset_begin;
    if (read_size == 0) {
        return; 
    }

    if (read_size > log_block_reader_info.blocksize) {
        read_size = ((log_block_reader_info.blocksize * 3) >> 2);
        begin = end - read_size;
    }

    u32 align_read_begin = (begin & (log_block_reader_info.blocksize - 1));

    iovec iov[4];
    if ((begin & (~(log_block_reader_info.blocksize - 1))) !=
                ((begin + read_size) & (~(log_block_reader_info.blocksize - 1)))) {

        char lenbuffer[8];
        *(int*)lenbuffer = (log_block_reader_info.blocksize - align_read_begin) + (sizeof(int) << 1);
        *(int*)(lenbuffer + sizeof(int)) = align_read_begin;
        iov[0].iov_base = lenbuffer;
        iov[0].iov_len = (sizeof(int) << 1);
        iov[1].iov_base = (char*)(log_block_reader_info.vaddr) + align_read_begin;
        iov[1].iov_len = log_block_reader_info.blocksize - align_read_begin;
        size_t len = iov[0].iov_len + iov[1].iov_len;

        int remain_size = read_size - iov[1].iov_len;
        char lenbuffer1[8];
        *(int*)lenbuffer1 = remain_size + (sizeof(int) << 1);
        *(int*)(lenbuffer1 + sizeof(int)) = 0;
        iov[2].iov_base = lenbuffer1;
        iov[2].iov_len = (sizeof(int) << 1);
        iov[3].iov_base = (char*)(log_block_reader_info.vaddr);
        iov[3].iov_len = remain_size; 
        size_t len1 = iov[2].iov_len + iov[3].iov_len;

        data_callback_func_(name, process_id, thread_id, iov, 4, len + len1);
    } else {

        char lenbuffer[8];
        *(int*)lenbuffer = read_size + (sizeof(int) << 1);
        *(int*)(lenbuffer + sizeof(int)) = align_read_begin;
        iov[0].iov_base = lenbuffer;
        iov[0].iov_len = (sizeof(int) << 1);
        iov[1].iov_base = (char*)(log_block_reader_info.vaddr) + align_read_begin;
        iov[1].iov_len = read_size; 
        size_t len = iov[0].iov_len + iov[1].iov_len;

        data_callback_func_(name, process_id, thread_id, iov, 2, len);
    }
}

bool LogBlockBatchConsumer::OpenLogBlockThreadHandle(LogBlockReaderThreadHandle &reader_thread_handle) {
    auto ret = OpenLogBlockReaderThreadHandle(&reader_thread_handle);
    if (ret == 0) {
        return true;
    }
    return false;
}

void LogBlockBatchConsumer::Stop() {
    running_.store(false);
}

void LogBlockBatchConsumer::Wait() {
    for (auto &th : threads_) {
        if (th.joinable()) {
            th.join();
        }
    }
}

} // namespace logblock
} // namespace netaos
} // namespace hozon
