// 
// Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
// 
/// @file hz_log_service_sink.hpp
/// @brief 
/// @author xumengjun@hozonauto.com
/// @version 1.0.0
/// @date 2023-11-07

#ifndef __ADAPTER_HZ_LOGSERVICE_SINK_H__
#define __ADAPTER_HZ_LOGSERVICE_SINK_H__

#include <string>

#include "spdlog/common.h"
#include "spdlog/formatter.h"

namespace hozon {
namespace netaos {
namespace log {

class HzLogServiceSink {
public:
    HzLogServiceSink(const std::string &file_base_name);
    ~HzLogServiceSink();

    void log(spdlog::level::level_enum level, const char *data, size_t size);
    void log(spdlog::level::level_enum level, const std::string &message);

    void set_pattern(const std::string &pattern);
    void set_level(spdlog::level::level_enum level);
    bool should_log(spdlog::level::level_enum level) const;

 private:
    std::string file_base_name_;
    std::unique_ptr<spdlog::formatter> formatter_;
    spdlog::level::level_enum level_;
};

}  // namespace log 
}  // namespace netaos 
}  // namespace hozon 

#endif // __ADAPTER_HZ_LOGSERVICE_SINK_H__
