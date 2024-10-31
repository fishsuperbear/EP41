/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
*
* @File: custom_time_util.cpp
* @Date: 2023/12/14
* @Author: cheng
* @Desc: --
*/

#include <iostream>
#include <sys/stat.h>

#include "config_param.h"
#include "utils/include/time_utils.h"
#include "utils/include/custom_time_util.h"

using namespace hozon::netaos::cfg;
namespace hozon {
namespace netaos {
namespace dc {

time_t CustomTimeUtil::getLocalTime() {
    struct timespec time = {0};
    auto cfgMgr = ConfigParam::Instance();
    cfgMgr->Init();
     int64_t value;
    auto res = cfgMgr->GetParam<int64_t>("time/mp_offset", value);
    if (CfgResultCode::CONFIG_OK!=res || value ==0) {
        clock_gettime(CLOCK_VIRTUAL, &time);
        return time.tv_sec;
    } else {
        clock_gettime(CLOCK_REALTIME, &time);
        return time.tv_sec + value;
    }
}

time_t CustomTimeUtil::getUnixTime() {
    return getLocalTime();
}

}  // namespace dc
}  // namespace netaos
}  // namespace hozon