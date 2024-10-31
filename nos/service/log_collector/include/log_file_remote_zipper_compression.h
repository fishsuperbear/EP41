// 
// Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
// 
/// @file log_file_remote_zipper_compression.h
/// @brief 
/// @author xumengjun@hozonauto.com
/// @version 1.0.0
/// @date 2023-11-15

#ifndef __LOG_COLLECTOR_INCLUDE_LOG_FILE_REMORE_ZIPPER_COMPRESSION_H__
#define __LOG_COLLECTOR_INCLUDE_LOG_FILE_REMORE_ZIPPER_COMPRESSION_H__

#include <string>
#include <memory>

#include "zmq_ipc/manager/zmq_ipc_client.h"

#include "log_collector/include/log_file_compression_base.h"

namespace hozon {
namespace netaos {
namespace logcollector {

class LogFileRemoteZipperCompression : public LogFileCompressionBase {
public:
    LogFileRemoteZipperCompression(const std::string &cmpr_log_service_name = "tcp://localhost:5778");
    ~LogFileRemoteZipperCompression();
public:
    bool DO(const std::string &appid, const std::string &file_path,
                const std::string &file_name, std::string &zip_result_file) override;

protected:
    std::string compression_log_service_name_ = "tcp://localhost:5778";
    std::unique_ptr<hozon::netaos::zmqipc::ZmqIpcClient> client_;
};

} // namespace logcollector
} // namespace netaos
} // namespace hozon

#endif // __LOG_COLLECTOR_INCLUDE_LOG_FILE_REMORE_ZIPPER_COMPRESSION_H__
