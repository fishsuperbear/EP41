// 
// Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
// 
/// @file log_block_consumer.h
/// @brief 
/// @author xumengjun@hozonauto.com
/// @version 1.0.0
/// @date 2023-11-07

#ifndef __NETAOS_MODULES_LOG_BLOCK_LOG_READER_H__
#define __NETAOS_MODULES_LOG_BLOCK_LOG_READER_H__

#include <functional>
#include <thread>
#include <atomic>
#include <vector>
#include <string>

#include "log_block_reader.h"
#include "logblock_helper/include/log_block_common_defs.h"

namespace hozon {
namespace netaos {
namespace logblock {

// /*******************************************************
/// @brief This interface encapsulates the function of getting logs from logblock.
// *******************************************************/
class LogBlockConsumer {
public:
    using CallBackFunc = std::function<void (const logblock::LogHeader *header, const logblock::LogBody *body)>;
    using ErrorCallBackFunc = std::function<void (const std::string &message)>;

public:
    static LogBlockConsumer& Instance();

public:
    // /*******************************************************
    /// @brief start log consumer 
    ///
    /// @param: threads_num
    /// @param: call_back_func
    ///
    /// @returns: true/false
    // *******************************************************/
    bool Start(int threads_num, const CallBackFunc &data_callback_func, 
                const ErrorCallBackFunc &error_callback_func = {});
    void Stop();

private:
    void Run();

    bool OpenLogBlockThreadHandle(LogBlockReaderThreadHandle &reader_thread_handle);
    bool UpdateLogBlockReaderInfo(LogBlockReaderThreadHandle thread_handle, 
                LogBlockReaderInfo &log_block_reader_info);

    void HandleData(LogBlockReaderInfo &log_block_reader_info);
    void ParseData(const char *data_addr, int start_parse_pos, int end_parse_pos);

private:
    LogBlockConsumer();
    ~LogBlockConsumer();

private:
    bool inited_ = false;
    int thread_num_ = 0;
    std::vector<std::thread> threads_;
    std::atomic<bool> running_ = {true};
    CallBackFunc data_callback_func_;
    ErrorCallBackFunc error_callback_func_;
};

} // namespace logblock
} // namespace netaos
} // namespace hozon

#endif // __NETAOS_MODULES_LOG_BLOCK_LOG_READER_H__
