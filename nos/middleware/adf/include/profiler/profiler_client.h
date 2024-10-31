#pragma once

#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace hozon {
namespace netaos {
namespace adf {
#define HEADER_NAME_TAG 0
#define ITEM_TAG 1
#define INSTANCE_NAME_TAG 2

#define PROFILER_SERVER_MULTICAST_ADDR "224.0.0.33"
#define CHECKPOINT_MULTICAST_PORT 23678
#define LATENCY_MULTICAST_PORT 23679

template <typename T>
class ProfilerClient {
   public:
    ~ProfilerClient() {
        if (_sock_fd >= 0) {
            // setsockopt(_sock_fd, IPPROTO_IP, IP_DROP_MEMBERSHIP, &_remote_addr, sizeof(_remote_addr));
            close(_sock_fd);
            _sock_fd = -1;
        }
    }

    bool Init(const std::string& name, const char* server_addr, uint16_t server_port,
              std::vector<std::string>& header_names) {
        _sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
        if (_sock_fd < 0) {
            return false;
        }

        memset(&_remote_addr, 0, sizeof(_remote_addr));
        _remote_addr.sin_family = AF_INET;
        _remote_addr.sin_port = htons(server_port);
        _remote_addr.sin_addr.s_addr = inet_addr(server_addr);

        // calc size
        uint32_t frame_size = 0;
        frame_size += 4;
        frame_size += name.size();
        for (auto header_name : header_names) {
            frame_size += header_name.size();
            frame_size += 4;  // tag:u16 len:u16
        }
        frame_size += 4;  // tag:u16 len:u16 double
        frame_size += sizeof(T) * (header_names.size());
        _frame.resize(frame_size);

        // set name header
        uint32_t index = 0;

        // instance name
        const uint16_t instance_name_tag = INSTANCE_NAME_TAG;
        memcpy(_frame.data() + index, &instance_name_tag, 2);
        index += 2;

        const uint16_t instance_name_size = name.size();
        memcpy(_frame.data() + index, &instance_name_size, 2);
        index += 2;

        memcpy(_frame.data() + index, name.data(), name.size());
        index += name.size();

        // checkpoint name
        for (auto header_name : header_names) {
            const uint16_t header_tag = HEADER_NAME_TAG;
            memcpy(_frame.data() + index, &header_tag, 2);
            index += 2;

            uint16_t cpt_name_size = header_name.size();
            memcpy(_frame.data() + index, &cpt_name_size, 2);
            index += 2;

            memcpy(_frame.data() + index, header_name.data(), header_name.size());
            index += header_name.size();
        }

        _header_size = index;
        return true;
    }

    void Send(const std::vector<T>& datas) {
        uint32_t index = _header_size;
        const uint16_t item_tag = ITEM_TAG;
        memcpy(_frame.data() + index, &item_tag, 2);
        index += 2;

        const uint16_t length = sizeof(T) * (datas.size());
        memcpy(_frame.data() + index, &length, 2);
        index += 2;

        for (auto data : datas) {
            memcpy(_frame.data() + index, &data, sizeof(T));
            index += sizeof(T);
        }

        if (sendto(_sock_fd, _frame.data(), _frame.size(), 0, (struct sockaddr*)&_remote_addr, sizeof(_remote_addr)) !=
            static_cast<long int>(_frame.size())) {
            std::cout << "fail to send, errno " << errno << "\n";
        }
    }

   private:
    int _sock_fd = -1;
    struct sockaddr_in _remote_addr;
    struct sockaddr_in _local_addr;
    struct ip_mreq _mreq;
    std::vector<uint8_t> _frame;
    uint32_t _header_size;
};
}  // namespace adf
}  // namespace netaos

}  // namespace hozon
