/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: manager.cpp
 * @Date: 2023/08/08
 * @Author: cheng
 * @Desc: --
 */


#include "manager/include/manager.h"

namespace hozon {
namespace netaos {
namespace dc {
void TsleepWithWakeup(const long long milliseconds,
                              std::atomic_bool &wakeupFlag,
                              const long long minIntervalMs) {
    if (milliseconds <= 0)
        return;
    long long sleep_count = milliseconds / minIntervalMs;

    while (!wakeupFlag.load(std::memory_order::memory_order_acquire) && (sleep_count-- > 0)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(minIntervalMs));
    }
}

void TsleepWithWakeup(const long long milliseconds, std::atomic_bool &wakeupFlag) {
    const long long minIntervalMs = 50;
    TsleepWithWakeup(milliseconds, wakeupFlag, minIntervalMs);
}


}  // namespace dc
}  // namespace netaos
}  // namespace hozon
