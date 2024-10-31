/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: doip client test manager
 */
#ifndef TEST_COMMON_H
#define TEST_COMMON_H


#include <sys/time.h>
#include <iostream>
#include <sstream> 
#include <mutex>

extern std::mutex mutex_log_debug;
extern void printfVecHex(const char *head, uint8_t *value, uint32_t size);
extern bool log_debug;

#define BIG_TO_LITTLE_ENDIAN_16(x) ((((x) & 0xff00) >> 8) | (((x) & 0x00ff) << 8))
#define LITTLE_TO_BIG_ENDIAN_16(x) ((((x) & 0xff00) >> 8) | (((x) & 0x00ff) << 8))
#define BIG_TO_LITTLE_ENDIAN_32(x) ((((x) & 0xff000000) >> 24) | (((x) & 0x00ff0000) >> 8) | (((x) & 0x0000ff00) << 8) | (((x) & 0x000000ff) << 24))
#define LITTLE_TO_BIG_ENDIAN_32(x) ((((x) & 0xff000000) >> 24) | (((x) & 0x00ff0000) >> 8) | (((x) & 0x0000ff00) << 8) | (((x) & 0x000000ff) << 24))


class DebugLogger {
public:
    DebugLogger(bool need_log) :
        _need_log(need_log) {
        buffer << log_Time();
    }

    ~DebugLogger() {
        if (_need_log) {
            std::lock_guard<std::mutex> lock(mutex_log_debug);
            std::cout << buffer.str() << std::endl;
        }
    }


    char* log_Time(void)
    {
        static char szTime[100];
        struct timeval tv;
        gettimeofday(&tv, NULL);

        struct tm tm_time;
        localtime_r(&tv.tv_sec, &tm_time);

        snprintf(szTime, sizeof(szTime)-1, "[%02d-%02d-%02d %02d:%02d:%02d.%06ld] ",
                tm_time.tm_year + 1900, tm_time.tm_mon + 1, tm_time.tm_mday, tm_time.tm_hour, tm_time.tm_min, tm_time.tm_sec, tv.tv_usec);
        return szTime;
    }


    template<typename T>
    DebugLogger& operator<<(const T& value) {
        if (_need_log) {
            buffer << value;
        }
        
        return *this;
    }

private:
    bool _need_log = false;
    std::string _head;
    std::stringstream buffer;
};

#define DEBUG_LOG           DebugLogger(log_debug) << "[  DEBUG   ] "
#define INFO_LOG            DebugLogger(true) << "\033[32m[  INFO    ]\033[0m "
#define FAIL_LOG            DebugLogger(true) << "\033[31m[  FAILED  ]\033[0m "
#define PASS_LOG            DebugLogger(true) << "\033[32m[  PASSED  ]\033[0m "



#endif

