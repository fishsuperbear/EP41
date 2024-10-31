#pragma once

#include <bits/time.h>
#include <bits/types/struct_timespec.h>

namespace hozon {
namespace netaos {
namespace extra {

double GetRealTimestamp() {
    struct timespec time_now;
    clock_gettime(CLOCK_REALTIME, &time_now);

    return (double)time_now.tv_sec + (double)time_now.tv_nsec / 1000 / 1000 / 1000;
}

}  // namespace extra
}  // namespace netaos
}  // namespace hozon