#pragma once

#include <cstdint>
#include <time.h>

namespace hozon {
namespace netaos {
namespace tsync {

struct UdpTsyncPacket {
    uint64_t t1;
    uint64_t t2;
    uint64_t t3;
    uint64_t t4;
    uint64_t seq;
};

#define MASTER_PORT     23679
#define BUFF_LEN        200

class Time {
public:
    static uint64_t GetCurrTimeStampUs() {
        struct timespec time;
        clock_gettime(CLOCK_REALTIME, &time);

        return time.tv_sec * 1000 * 1000 + time.tv_nsec / 1000;
    }

    static void SetTimeWithDiff(int64_t delta_us) {
        uint64_t target_time = GetCurrTimeStampUs() + delta_us;

        struct timespec time;
        time.tv_sec = target_time / 1000 / 1000;
        time.tv_nsec = (target_time - time.tv_sec * 1000 * 1000) * 1000;
        clock_settime(CLOCK_REALTIME, &time);
    }

    static void SetTime(uint64_t times_us) {
        struct timespec time;
        time.tv_sec = times_us / 1000 / 1000;
        time.tv_nsec = (times_us - time.tv_sec * 1000 * 1000) * 1000;
        clock_settime(CLOCK_REALTIME, &time);
    }
};

}
}
}