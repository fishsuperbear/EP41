// 
// Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
// 
/// @file hz_log_service_sink.cpp
/// @brief 
/// @author xumengjun@hozonauto.com
/// @version 1.0.0
/// @date 2023-11-07

#include "adapter/hz_log_service_sink.hpp" 

#include <time.h>

#include "spdlog/pattern_formatter.h"

#ifdef BUILD_FOR_ORIN
#include "logblock_helper/include/log_block_producer.h"
#include "logblock_helper/include/log_block_common_defs.h"
#endif

namespace hozon {
namespace netaos {
namespace log {

HzLogServiceSink::HzLogServiceSink(const std::string &file_base_name) :
    file_base_name_(file_base_name) {
}

HzLogServiceSink::~HzLogServiceSink() {
}

void HzLogServiceSink::log(spdlog::level::level_enum level, const char* data, size_t size) {
    if (!should_log(level)) {
        return;
    }

    spdlog::source_loc source;
    spdlog::details::log_msg msg(source,
                "",
                level,
                spdlog::string_view_t(data, size));

    spdlog::memory_buf_t formatted;
    formatter_->format(msg, formatted);

#ifdef BUILD_FOR_ORIN
    hozon::netaos::logblock::LogBlockProducer::Instance().Write(file_base_name_,
                            formatted.data(), formatted.size());
#endif
}

void HzLogServiceSink::log(spdlog::level::level_enum level, const std::string &message) { 
    log(level, message.data(), message.size());
}

void HzLogServiceSink::set_pattern(const std::string &pattern) {
    formatter_.reset(new spdlog::pattern_formatter(pattern));
}

void HzLogServiceSink::set_level(spdlog::level::level_enum level) {
    level_ = level;
}

bool HzLogServiceSink::should_log(spdlog::level::level_enum level) const {
  return level >= level_;
}

}  // namespace logserver
}  // namespace framework
}  // namespace netaos
