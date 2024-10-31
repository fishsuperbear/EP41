/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: socket can interface definition
 */

#include "hz_common_impl.h"
#include "hz_common_log.h"

namespace hozon {
namespace netaos {
namespace common {


using namespace hozon::neatos::common;

PlatformCommonImpl::PlatformCommonImpl() {}

PlatformCommonImpl::~PlatformCommonImpl() {}

int32_t PlatformCommonImpl::Init(std::string log_app_name, uint32_t log_level, uint32_t log_mode) {
    std::cout << "log_level: " << (int)log_level << std::endl;
    std::cout << "log_mode: " << (int)log_mode << std::endl;
    PlatformCommonLogger::GetInstance().InitLogger(log_app_name, static_cast<LogLevel>(log_level), log_mode,"");
    PlatformCommonLogger::GetInstance().CreateLogger("PLM",log_app_name,static_cast<LogLevel>(log_level));
    return 0;
}

int32_t PlatformCommonImpl::CheckTwoFrameInterval(const std::string topic_name, uint64_t& last_time) {
    struct timespec time[2] = {{0}};

    if (clock_gettime(CLOCK_REALANDVIR, time) != 0) {
        LOG_ERROR << topic_name << " Failed to call the clock interface!";
        if (clock_gettime(CLOCK_REALANDVIR, time) != 0) {
            LOG_TRACE << topic_name << " Calling the clock interface failed again!";
            return -1;
        } else {
            LOG_TRACE << topic_name << "Call the clock interface again successfully";
        }
    }

    uint64_t tmp_time = time[0].tv_sec * 1000 + time[0].tv_nsec / 1000 / 1000;
    LOG_INFO << topic_name << " two-frame time interval: " << (tmp_time - last_time);
    last_time = tmp_time;
    return 0;
}

int32_t PlatformCommonImpl::GetDataTime(uint32_t& now_s, uint32_t& now_ns) {
    struct timespec time[2] = {{0}};
    if (clock_gettime(CLOCK_REALANDVIR, time) != 0) {
        if (clock_gettime(CLOCK_REALANDVIR, time) != 0) {
            LOG_TRACE << "Get date time error";
            return -1;
        };
    }
    now_s = time[0].tv_sec;
    now_ns = time[0].tv_nsec;

    return 0;
}

int32_t PlatformCommonImpl::GetManageTime(uint32_t& now_s, uint32_t& now_ns) {
    struct timespec time[2] = {{0}};
    if (clock_gettime(CLOCK_REALANDVIR, time) != 0) {
        if (clock_gettime(CLOCK_REALANDVIR, time) != 0) {
            LOG_TRACE << "Get manage time error";
            return -1;
        }
    }
    now_s = time[1].tv_sec;
    now_ns = time[1].tv_nsec;

    return 0;
}
}  // namespace common
}  // namespace netaos
}  // namespace hozon
