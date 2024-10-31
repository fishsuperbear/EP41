/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
 * Create: 2021/12/9.
 */

#ifndef ARA_LOG_LOGGER_FLOWCTRL_H
#define ARA_LOG_LOGGER_FLOWCTRL_H
#include <atomic>
#include <chrono>
#include "ara/log/logging.h"

#define LOGFATAL    LogFatal
#define LOGERROR    LogError
#define LOGWARN     LogWarn
#define LOGDEBUG    LogDebug
#define LOGINFO     LogInfo
#define LOGVERBOSE  LogVerbose

#define LOG_EVERY_N_LINE_0(base, line) base ## line
#define LOG_EVERY_N_LINE(base, line) LOG_EVERY_N_LINE_0(base, line)
#define START_PARAM LOG_EVERY_N_LINE(_log_interval_start, __LINE__)
#define END_PARAM LOG_EVERY_N_LINE(_log_interval_end, __LINE__)
#define IS_FIRST_VALUE LOG_EVERY_N_LINE(_log_is_first, __LINE__)
#define SPENT_VALUE LOG_EVERY_N_LINE(_log_interval_spent, __LINE__)
#define SPENT_TIME_VALUE LOG_EVERY_N_LINE(_log_interval_spent_time, __LINE__)
#define TIMES_PARAM LOG_EVERY_N_LINE(_log_times_param, __LINE__)
#define EVENRY_LOGGER LOG_EVERY_N_LINE(_every_log, __LINE__)

#define LogEveryT(level, interval) \
    static std::chrono::steady_clock::time_point START_PARAM = std::chrono::steady_clock::now(); \
    const std::chrono::steady_clock::time_point END_PARAM = std::chrono::steady_clock::now(); \
    static std::atomic<bool> IS_FIRST_VALUE(true); \
    const std::chrono::duration<double, std::milli> SPENT_VALUE = END_PARAM - START_PARAM; \
    uint32_t SPENT_TIME_VALUE = static_cast<uint32_t>(SPENT_VALUE.count()); \
    if (SPENT_TIME_VALUE >= interval) { \
        START_PARAM = END_PARAM; \
    } \
    if (IS_FIRST_VALUE.load()) { \
        IS_FIRST_VALUE.store(false); \
        SPENT_TIME_VALUE = interval; \
    } \
    static ara::log::Logger& EVENRY_LOGGER = CreateLogger("ERYT","global_logger",ara::log::LogLevel::kVerbose); \
    if (SPENT_TIME_VALUE >= interval) \
        EVENRY_LOGGER.level()

#define LogEveryN(level, times) \
    static std::atomic<uint32_t> TIMES_PARAM(0U); \
    TIMES_PARAM = TIMES_PARAM % times; \
    TIMES_PARAM++; \
    static ara::log::Logger& EVENRY_LOGGER = CreateLogger("ERYN","global_logger",ara::log::LogLevel::kVerbose); \
    if (TIMES_PARAM == 1U) \
        EVENRY_LOGGER.level()

#endif