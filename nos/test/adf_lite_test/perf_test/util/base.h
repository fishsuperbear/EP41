#pragma once
#include <cstdint>
#include <memory>


inline uint64_t GetRealTimestamp_us() {
    struct timespec time_now;
    clock_gettime(CLOCK_REALTIME, &time_now);

    return time_now.tv_sec * 1000000 + time_now.tv_nsec / 1000;
}
