/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: time_utils.cpp
 * @Date: 2023/08/15
 * @Author: cheng
 * @Desc: --
 */

#include "utils/include/time_utils.h"
//#include <sys_ctr.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <iostream>
#include <map>

namespace hozon {
namespace netaos {
namespace dc {

std::string TimeUtils::GET_TIMER_USE_PLANE = "MP";
int32_t TimeUtils::TIME_OFFSET_SECOND = 0;
void TimeUtils::sleepWithWakeup(const long long milliseconds,
                              std::atomic_bool &wakeupFlag,
                              const long long minIntervalMs) {
    if (milliseconds <= 0)
        return;
    long long sleep_count = milliseconds / minIntervalMs;
    long long sleepMin = milliseconds % minIntervalMs;
    if (!wakeupFlag.load(std::memory_order::memory_order_acquire) && sleepMin >0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds % minIntervalMs));
    }
    while (!wakeupFlag.load(std::memory_order::memory_order_acquire) && (sleep_count-- > 0)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(minIntervalMs));
    }
}

void TimeUtils::sleepWithWakeup(const long long milliseconds, std::atomic_bool& wakeupFlag) {
    const long long minIntervalMs = 50;
    sleepWithWakeup(milliseconds, wakeupFlag, minIntervalMs);
}

void TimeUtils::sleep(const long long milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

uint64_t TimeUtils::getDataTimeMicroSec() {
    struct timespec time = {0};
    clock_gettime(CLOCK_REALTIME, &time);
    uint64_t micro_secs = time.tv_sec * 1000000 + time.tv_nsec / 1000;

    return micro_secs;
}

uint64_t TimeUtils::getMgmtTimeMicroSec() {
    struct timespec time = {0};
    clock_gettime(CLOCK_VIRTUAL, &time);
    uint64_t micro_secs = time.tv_sec * 1000000 + time.tv_nsec / 1000;

    return micro_secs;
}

uint32_t TimeUtils::getDataTimeSec() {
    struct timespec time = {0};
    clock_gettime(CLOCK_REALTIME, &time);
    return static_cast<uint32_t>(time.tv_sec);
}

uint32_t TimeUtils::getMgmtTimeSec() {
    struct timespec time = {0};
    clock_gettime(CLOCK_VIRTUAL, &time);
    return static_cast<uint32_t>(time.tv_sec);
}

uint32_t TimeUtils::getDataTimeMicroSecInSec() {
    struct timespec time = {0};
    clock_gettime(CLOCK_REALTIME, &time);
    return static_cast<uint32_t>(time.tv_nsec / 1000);
}

uint32_t TimeUtils::getMgmtTimeMicroSecInSec() {
    struct timespec time = {0};
    clock_gettime(CLOCK_VIRTUAL, &time);
    return static_cast<uint32_t>(time.tv_nsec / 1000);
}

Timestamp TimeUtils::getMgmtTimestamp() {
    struct timespec time = {0};
    clock_gettime(CLOCK_VIRTUAL, &time);

    Timestamp timestamp {static_cast<uint32_t>(time.tv_nsec), static_cast<uint32_t>(time.tv_sec)};
    return timestamp;
}

Timestamp TimeUtils::getDataTimestamp() {
    struct timespec time = {0};
    clock_gettime(CLOCK_REALTIME, &time);

    Timestamp timestamp {static_cast<uint32_t>(time.tv_nsec), static_cast<uint32_t>(time.tv_sec)};
    return timestamp;
}

uint64_t TimeUtils::getMgmtTime() {
    struct timespec time = {0};
    clock_gettime(CLOCK_VIRTUAL, &time);

    Timestamp timestamp {static_cast<uint32_t>(time.tv_nsec), static_cast<uint32_t>(time.tv_sec)};
    return *reinterpret_cast<uint64_t*>(&timestamp);
}

uint64_t TimeUtils::getDataTime() {
    struct timespec time = {0};
    clock_gettime(CLOCK_REALTIME, &time);

    Timestamp timestamp {static_cast<uint32_t>(time.tv_nsec), static_cast<uint32_t>(time.tv_sec)};
    return *reinterpret_cast<uint64_t*>(&timestamp);
}

std::string TimeUtils::sec2ReadableStr(uint32_t sec) {
    time_t time = sec;
    struct tm* timeinfo = localtime(&time);
    char buf[32] = {0};
    snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d",
             (timeinfo->tm_year + 1900)%10000,
             (timeinfo->tm_mon + 1)%100,
             (timeinfo->tm_mday)%100,
             timeinfo->tm_hour%100,
             timeinfo->tm_min%100,
             timeinfo->tm_sec%100);

    return std::string(buf);
}

std::string TimeUtils::timestamp2ReadableStr(Timestamp timestamp) {
    time_t time = timestamp.sec;
    struct tm* timeinfo = localtime(&time);
    char buf[35] = {0};
    snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d.%06d",
             (timeinfo->tm_year + 1900)%10000,
             (timeinfo->tm_mon + 1)%100,
             timeinfo->tm_mday%100,
             timeinfo->tm_hour%100,
             timeinfo->tm_min%100,
             timeinfo->tm_sec%100,
             (timestamp.nsec / 1000));

    return std::string(buf);
}
//
//std::string TimeUtils::getAppDirectory() {
//    char buf[1024] = {0};
//    if (!getcwd(buf, sizeof(buf))) {
//        std::cout << "Can not get working directory.\n";
//        return "";
//    }
//    std::string wk(buf);
//    if (wk.size() <= 0) {
//        std::cout << "Can not get working directory.\n";
//        return "";
//    }
//
//    if (wk[wk.size() - 1] != '/') {
//        wk = wk + '/';
//    }
//
//    std::string temp_bin_dir = wk + "bin";
//
//    struct stat bin_dir_stat;
//    if (stat(temp_bin_dir.c_str(), &bin_dir_stat) == 0) {
//        if (S_ISDIR(bin_dir_stat.st_mode)) {
//            return wk;
//        }
//    }
//
//    return APP_DIR_DATA_COLLECT;
//}

std::string TimeUtils::convertTime2ReadableStr(uint32_t sec) {
    struct tm timeinfo = {0};
    time_t temp = sec;
    localtime_r(&temp, &timeinfo);
    char time_buf[128] = {0};

    snprintf(time_buf, sizeof(time_buf) - 1, "%04d/%02d/%02d %02d:%02d:%02d", timeinfo.tm_year + 1900, timeinfo.tm_mon + 1, timeinfo.tm_mday, timeinfo.tm_hour, timeinfo.tm_min, timeinfo.tm_sec);
    return std::string(time_buf);
}

void TimeUtils::SetThreadName(std::string name) {

    if (name.size() > 16) {
        name = name.substr(name.size() - 15, 15);
    }
    pthread_setname_np(pthread_self(), name.c_str());
}

std::string TimeUtils::formatTimeStrForFileName(time_t unix_time) {
    struct tm timeinfo = {0};
    localtime_r(&unix_time, &timeinfo);
    char time_buf[128] = {0};

    snprintf(time_buf, sizeof(time_buf) - 1, "%04d%02d%02d-%02d%02d%02d",
             (timeinfo.tm_year + 1900)%10000,
             (timeinfo.tm_mon + 1)%100,
             timeinfo.tm_mday%100,
             timeinfo.tm_hour%100,
             timeinfo.tm_min%100,
             timeinfo.tm_sec%100);

    return std::string(time_buf);
}

std::string TimeUtils::formatTimeStrForFileName(Timestamp timestamp) {
    time_t time = timestamp.sec;
    struct tm *timeinfo = localtime(&time);
    char buf[35] = {0};
    snprintf(buf, sizeof(buf), "%04d-%02d-%02d-%02d-%02d-%02d-%06d",
             (timeinfo->tm_year + 1900)%10000,
             (timeinfo->tm_mon + 1)%100,
             (timeinfo->tm_mday)%100,
             (timeinfo->tm_hour)%100,
             (timeinfo->tm_min)%100,
             (timeinfo->tm_sec)%100,
             (timestamp.nsec / 1000));

    return std::string(buf);
}



}  // namespace dc
}  // namespace netaos
}  // namespace hozon
