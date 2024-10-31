/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: 可视化内置结构：Times
 */

#ifndef VIZ_TIMES_H
#define VIZ_TIMES_H

#include <sys/time.h>
#include <cstdint>

namespace mdc {
namespace visual {
struct Times {
    uint32_t sec;
    uint32_t nsec;
    Times() : sec(0U), nsec(0U) {}
    Times(const uint32_t& vSec, const uint32_t& vNsec) : sec(vSec), nsec(vNsec) {}
    static Times now()
    {
        struct timeval ctime;
        (void)gettimeofday(&ctime, nullptr);
        Times t;
        t.sec = static_cast<uint32_t>(ctime.tv_sec);
        t.nsec = static_cast<uint32_t>(ctime.tv_usec) * 1000U; // usec * 1000 -> nsec
        return t;
    }
};
}
}
#endif // VIZ_TIMES_H
