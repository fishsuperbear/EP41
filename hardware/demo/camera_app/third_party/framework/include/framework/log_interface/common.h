#pragma once

#include "framework/common/macros.h"
#include "framework/log_interface/level.h"
#include "spdlog/spdlog.h"

// always inline
#ifdef _MSC_VER
#define SPDLOG_ALWAYS_INLINE __forceinline
#elif __GNUC__ >= 3
#define SPDLOG_ALWAYS_INLINE inline __attribute__((__always_inline__))
#else
#define SPDLOG_ALWAYS_INLINE inline
#endif

using netaos::framework::LogLevel;
using spdlog::level::level_enum;

namespace netaos {
namespace framework {
namespace loginterface {

inline level_enum netaos_to_spdlog_level(LogLevel level) {
  level_enum spd_level;

  switch (level) {
    case LogLevel::DEBUG: {
      spd_level = level_enum::debug;
      break;
    }
    case LogLevel::INFO: {
      spd_level = level_enum::info;
      break;
    }
    case LogLevel::WARNING: {
      spd_level = level_enum::warn;
      break;
    }
    case LogLevel::ERROR: {
      spd_level = level_enum::err;
      break;
    }
    case LogLevel::FATAL: {
      spd_level = level_enum::critical;
      break;
    }
    default:  // unknown treat as error, for not lack any messages
    {
      spd_level = level_enum::err;
      break;
    }
  }

  return spd_level;
}

}  // namespace loginterface
}  // namespace framework
}  // namespace netaos