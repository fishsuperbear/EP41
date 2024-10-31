#pragma once
#include <netinet/in.h>
#include <sys/socket.h>
#include <iostream>
#include <thread>
#include <yaml-cpp/yaml.h>
#include "hz_time_logger.h"

#define CLOCK_VIRTUAL (12)
#define MASTER_PORT (23458)
#define BUFF_LEN (1024)
#define NOT_SYNCED (0)
#define SYNCED (1)

namespace hozon {
namespace netaos {
namespace hz_time {

#pragma pack(2)

struct McuUtcMsg {
    uint8_t time_base_status;
    uint32_t nanoseconds;
    uint32_t seconds;
    uint16_t seconds_hi;
};

#pragma pack()

class Time {
   public:
    static Time& GetInstance() {
        static Time instance;
        return instance;
    }

    void SetParameter(const std::string& server_ip) { server_ip_ = server_ip; }

    int32_t Init();

    void Routine();

    int32_t Deinit();

    ~Time() = default;

   private:
    Time() = default;
    void IsSyncManageTime(const McuUtcMsg* recv_packet);

    int32_t socket_fd_;
    sockaddr_in socket_addr_;
    std::shared_ptr<std::thread> thread_;
    bool stop_flag_ = false;
    bool is_sync_ = false;
    uint8_t manual_sync_ = false;
    std::string server_ip_;
    uint64_t seq_ = 0;
};
}  // namespace hz_time
}  // namespace netaos
}  // namespace hozon