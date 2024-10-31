// 
// Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
// 
/// @file log_block_batch_consumer.h
/// @brief 
/// @author xumengjun@hozonauto.com
/// @version 1.0.0
/// @date 2023-11-06

#ifndef __NETAOS_MODULES_LOG_BLOCK_BATCH_CONSUMER_H___
#define __NETAOS_MODULES_LOG_BLOCK_BATCH_CONSUMER_H__

#include <functional>
#include <thread>
#include <atomic>
#include <vector>
#include <string>
#include <sys/uio.h>

#include "log_block_reader.h"

namespace hozon {
namespace netaos {
namespace logblock {

// /*******************************************************
/// @brief This interface encapsulates the function of getting logs from logblock.
// *******************************************************/
class LogBlockBatchConsumer {
public:
    using CallBackFunc = std::function<void (const char *name, unsigned int process_id,
                                            unsigned int thread_id,
                                            const iovec *vec, size_t count, size_t len)>;

public:
    static LogBlockBatchConsumer& Instance();

public:
    // /*******************************************************
    /// @brief start log consumer 
    ///
    /// @param: threads_num
    /// @param: call_back_func
    ///
    /// @returns: true/false
    // *******************************************************/
    bool Start(int threads_num, const CallBackFunc &data_callback_func);
    void Stop();
    void Wait();

private:
    void Run();

    bool OpenLogBlockThreadHandle(LogBlockReaderThreadHandle &reader_thread_handle);
    void HandleData(LogBlockReaderInfo &log_block_reader_info);

private:
    LogBlockBatchConsumer();
    ~LogBlockBatchConsumer();

private:
    bool inited_ = false;
    int thread_num_ = 0;
    std::vector<std::thread> threads_;
    std::atomic<bool> running_ = {true};
    CallBackFunc data_callback_func_;
};

} // namespace logblock
} // namespace netaos
} // namespace hozon

#endif // __NETAOS_MODULES_LOG_BLOCK_BATCH_CONSUMER_H__
