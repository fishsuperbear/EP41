#include "hz_time.h"
#include <arpa/inet.h>
#include <time.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include "hz_time_logger.h"
#include "config_param.h"

namespace hozon {
namespace netaos {
namespace hz_time {

auto config_server = hozon::netaos::cfg::ConfigParam::Instance();

int32_t Time::Init() {
    socket_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd_ < 0) {
        TIME_LOG_ERROR << "Fail to create socket, errno:" << errno;
        return -1;
    }

    memset(&socket_addr_, 0, sizeof(socket_addr_));
    socket_addr_.sin_family = AF_INET;
    socket_addr_.sin_addr.s_addr = inet_addr(server_ip_.c_str());
    socket_addr_.sin_port = htons(MASTER_PORT);

    if (bind(socket_fd_, (struct sockaddr*)&socket_addr_, sizeof(socket_addr_)) < 0) {
        TIME_LOG_ERROR << "Fail to bind socket, errno:" << errno;
        close(socket_fd_);
        return -1;
    }

    thread_ = std::make_shared<std::thread>(&Time::Routine, this);

    config_server->Init(2000);
    config_server->MonitorParam<uint8_t>(
        "time/manual_sync", [this](const std::string& client_name, const std::string& key, const uint8_t& manual_sync) {
            config_server->GetParam("time/manual_sync", manual_sync_);
            TIME_LOG_INFO << "Set manual sync:" << (int32_t)manual_sync;
        });
    config_server->GetParam("time/manual_sync", manual_sync_);

    return 0;
}

void Time::Routine() {
    if (!is_sync_) {
        config_server->SetParam<uint8_t>("time/time_synchronized", NOT_SYNCED);
    }

    char buf[BUFF_LEN];
    struct sockaddr_in client_addr;
    socklen_t len = sizeof(client_addr);
    int32_t count;

    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(socket_fd_, &rfds);
    while (!stop_flag_) {
        FD_ZERO(&rfds);
        FD_SET(socket_fd_, &rfds);
        memset(buf, 0, sizeof(buf));

        timeval timeout;
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;

        int ret = select(socket_fd_ + 1, &rfds, nullptr, nullptr, &timeout);
        if (ret == -1) {
            TIME_LOG_ERROR << "Fail to listen on interface, errno:" << errno;
            seq_++;
            continue;
        } else if (ret == 0) {
            TIME_LOG_ERROR << "Fail to recv msg";
            seq_++;
            continue;
        } else {
            if (FD_ISSET(socket_fd_, &rfds)) {
                count = recvfrom(socket_fd_, buf, BUFF_LEN, 0, (struct sockaddr*)&client_addr, &len);
                if (count != sizeof(McuUtcMsg)) {
                    TIME_LOG_ERROR << "Invalid recv size:" << count << ", expect size:" << len;
                    std::this_thread::sleep_for(std::chrono::microseconds(1000));
                    continue;
                }

                McuUtcMsg* recv_packet = (McuUtcMsg*)(&buf);
                TIME_LOG_INFO << "Recv time base status[" << seq_ << "]:" << std::to_string(recv_packet->time_base_status);
                TIME_LOG_INFO << "Recv time seconds[" << seq_ << "]:" << std::to_string(recv_packet->seconds);
                TIME_LOG_INFO << "Recv time nanoseconds[" << seq_ << "]:" << std::to_string(recv_packet->nanoseconds);

                IsSyncManageTime(recv_packet);
                seq_++;
            }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }
}

void Time::IsSyncManageTime(const McuUtcMsg* recv_packet) {
    double seconds = recv_packet->seconds;
    double nanoseconds = recv_packet->nanoseconds;
    auto recv_time_s = seconds + nanoseconds / 1000 / 1000 / 1000;
    struct timespec time_now = {0};

    if (0 != clock_gettime(CLOCK_VIRTUAL, &time_now)) {
        TIME_LOG_ERROR << "Fail to clock_gettime";
    }

    auto local_time_s = time_now.tv_sec + time_now.tv_nsec / 1000 / 1000 / 1000;
    if (std::abs(local_time_s - recv_time_s) > 1 && (SYNCED != manual_sync_)) {
        TIME_LOG_INFO << "Manage time need to sync";
        TIME_LOG_INFO << "Local manage time:" << std::to_string(local_time_s);
        TIME_LOG_INFO << "Recv manage time:" << std::to_string(recv_time_s);
        TIME_LOG_INFO << "Diff time:" << std::to_string(std::abs(local_time_s - recv_time_s));

        struct timespec time_recv = {0};
        time_recv.tv_sec = recv_packet->seconds;
        time_recv.tv_nsec = recv_packet->nanoseconds;
        if (0 == clock_settime(CLOCK_VIRTUAL, &time_recv)) {
            TIME_LOG_INFO << "Clock_settime successful!,time_recv: " << std::to_string(double(time_recv.tv_sec) + double(time_recv.tv_nsec));
        } else {
            TIME_LOG_ERROR << "Fail to clock_settime";
        }
    } else {
        if (!is_sync_ && recv_packet->time_base_status) {
            config_server->SetParam<uint8_t>("time/time_synchronized", SYNCED);
            TIME_LOG_INFO << "Set Param time/time_synchronized to:" << SYNCED;
            is_sync_ = true;
        }
        TIME_LOG_INFO << "Don't need to sync time";
    }
}

int32_t Time::Deinit() {
    stop_flag_ = true;
    if (thread_->joinable()) {
        thread_->join();
    }
    close(socket_fd_);

    return config_server->DeInit();
}
}  // namespace hz_time
}  // namespace netaos
}  // namespace hozon