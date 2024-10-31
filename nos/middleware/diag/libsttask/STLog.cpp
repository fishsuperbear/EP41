/**
 * Copyright @ 2021 - 2023 Hozon Auto Co., Ltd.
 * All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are NOT permitted except as agreed by
 * Hozon Auto Co., Ltd.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
/**
 * @file STLog.cpp
 * @brief implements of STLog
 */

#ifndef STLOG_H
#   include "STLog.h"
#endif

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sys/syscall.h>

#define ST_LOG_OPTION_SHORT   0
#define ST_LOG_OPTION_SUBTAG  1

#define LOG_SINGLE_LOG_SIZE   1024

namespace hozon {
namespace netaos {
namespace sttask {

#if ST_LOG_OPTION_SHORT
#   if ST_LOG_OPTION_SUBTAG
static const char* const ST_LOG_FOMAT_STR = "[%s][%s@%s(%d)] %s";
#   else
static const char* const ST_LOG_FOMAT_STR = "[%s@%s(%d)] %s";
#   endif
#else
#   if ST_LOG_OPTION_SUBTAG
static const char* const ST_LOG_FOMAT_STR = "%s %s %s/%c  [%d %ld %s@%s(%d) | %s]\n";
#   else
static const char* const ST_LOG_FOMAT_STR = "<tid:%5d> [%s@%s(%d)] %s";
#   endif
#endif


void STLog::output(const char* tag,
                   const char *subtag,
                   uint8_t type,
                    const char* func,
                   const char* file,
                   uint32_t line,
                   const char* format,
                   ...)
{
    char templog[LOG_SINGLE_LOG_SIZE] = { 0 };
    va_list list;
    va_start(list, format);
    vsnprintf(templog, LOG_SINGLE_LOG_SIZE - 1, format, list);
    va_end(list);

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm* timeinfo = localtime(&ts.tv_sec);
    char time_buf[128] = { 0 };
    snprintf(time_buf, sizeof(time_buf) - 1, "[%04d-%02d-%02d %02d:%02d:%02d.%03ld] ",
    timeinfo->tm_year + 1900,
    timeinfo->tm_mon + 1,
    timeinfo->tm_mday,
    timeinfo->tm_hour,
    timeinfo->tm_min,
    timeinfo->tm_sec,
    ts.tv_nsec/1000000);

    const char* filename = strrchr(file, '/');
    if (nullptr != filename) {
        filename = filename + 1;
    }
    else {
        filename = file;
    }

    printf(ST_LOG_FOMAT_STR, time_buf, tag, subtag, type, getpid(), (long int)syscall(__NR_gettid), func, filename, line, templog);
}

} // end of sttask
} // end of netaos
} // end of hozon
/* EOF */