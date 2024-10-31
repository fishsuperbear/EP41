#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <string>
#include <cstdint>
#include <unistd.h>
#include <thread>
#include "can_tsync_center/can_tsync_config_parser.h"

namespace hozon {
namespace netaos {
namespace tsync {

class CanTsync {
public:
    CanTsync();
    ~CanTsync();

    int32_t Start(const CanTSyncConfig& config);
    void Stop();

private:
    void CloseSocket();
    int32_t OpenSocket();
    void TSyncRoutine();
    int32_t SendSync(timespec& sync_time, uint8_t seq_num, uint64_t index);
    int32_t RecvSyncAck(timespec& ack_time, uint64_t index);
    int32_t SendFup(uint64_t time_diff_ns, uint8_t seq_num);
    uint8_t CalcCRC(uint8_t data[], uint8_t seq_num);

    CanTSyncConfig _config;
    int _socket = -1;
    // struct sockaddr_can _addr; // not used
    std::shared_ptr<std::thread> _thread;
    bool _need_stop = false;
};

}
}
}