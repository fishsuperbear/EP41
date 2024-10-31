#pragma once

#include <cstdint>
#include "proto/common/header.pb.h"

namespace hozon {
namespace netaos {
namespace adf {

inline uint64_t TimestampToUs(uint64_t sec, uint64_t nsec) {
    return sec * 1000 * 1000 + nsec / 1000;
}

inline uint64_t TimestampToUs(double d_sec) {
    return (uint64_t)(d_sec * 1000 * 1000);
}

inline double GetRealTimestamp() {
    struct timespec time_now;
    clock_gettime(CLOCK_REALTIME, &time_now);

    return (double)time_now.tv_sec + (double)time_now.tv_nsec / 1000 / 1000 / 1000;
}

}  // namespace adf
}  // namespace netaos
}  // namespace hozon