#include <cstdint>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <thread>
#include <memory>
#include "adf-lite/include/sig_stop.h"
#include "service/simple_udp_tsync/udp_tsync_packet.h"
#include "adf-lite/include/adf_lite_internal_logger.h"

namespace hozon {
namespace netaos {
namespace tsync {

#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>


void udp_msg_sender(int fd, struct sockaddr* dst)
{

    socklen_t len;
    struct sockaddr_in src;
    while(1)
    {
        char buf[BUFF_LEN] = "TEST UDP MSG!\n";
        len = sizeof(*dst);
        printf("client:%s\n",buf);  //打印自己发送的信息
        sendto(fd, buf, BUFF_LEN, 0, dst, len);
        memset(buf, 0, BUFF_LEN);
        recvfrom(fd, buf, BUFF_LEN, 0, (struct sockaddr*)&src, &len);  //接收来自server的信息
        printf("server:%s\n",buf);
        sleep(1);  //一秒发送一次消息
    }
}


class UdpTsyncSlave {
public:
    int32_t Init(const std::string& server_ip) {
        _socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
        if (_socket_fd < 0) {
            ADF_EARLY_LOG << "Fail to create socket.";
            return -1;
        }

        memset(&_ser_addr, 0, sizeof(_ser_addr));
        _ser_addr.sin_family = AF_INET;
        _ser_addr.sin_addr.s_addr = inet_addr(server_ip.c_str());
        // _ser_addr.sin_addr.s_addr = htonl(INADDR_ANY);
        _ser_addr.sin_port = htons(MASTER_PORT);

        _thread = std::make_shared<std::thread>(&UdpTsyncSlave::Routine, this);

        return 0;
    }

    ~UdpTsyncSlave() {
        if (_thread) {
            _running = false;
            _thread->join();
        }

        if (_socket_fd != 0) {
            close(_socket_fd);
            _socket_fd = 0;
        }
    }

    void Routine() {
        char buf[BUFF_LEN];
        socklen_t len = sizeof(_ser_addr);
        int32_t count;
        struct sockaddr_in client_addr;

        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(_socket_fd, &rfds);
        while (_running) {
            FD_ZERO(&rfds);
            FD_SET(_socket_fd, &rfds);

            memset(buf, 0, sizeof(buf));
            UdpTsyncPacket send_packet = {0};

            send_packet.seq = _seq;
            ADF_EARLY_LOG << "Send seq " << _seq;
            send_packet.t1 = Time::GetCurrTimeStampUs();
            int ret = sendto(_socket_fd, &send_packet, sizeof(UdpTsyncPacket), 0, (struct sockaddr*)&_ser_addr, len);
            ADF_EARLY_LOG << "send ret " << ret;

            timeval timeout;
            timeout.tv_sec = 1;
            timeout.tv_usec = 0;

            ret = select(_socket_fd + 1, &rfds, NULL, NULL, &timeout);
            if (ret == -1) {
                ADF_EARLY_LOG << "Fail to listen on interface, " << errno;
                ++_seq;
                continue;
            } 
            else if (ret == 0) {
                ADF_EARLY_LOG << "Fail to recv seq " << _seq;
                ++_seq;
                continue;
            } 
            else {
                count = recvfrom(_socket_fd, buf, BUFF_LEN, 0, (struct sockaddr*)&client_addr, &len);
                if (count != sizeof(UdpTsyncPacket)) {
                    ADF_EARLY_LOG << "Invalid size " << len;
                    ++_seq;
                    sleep(1);
                    continue;
                }

                UdpTsyncPacket* recv_packet = (UdpTsyncPacket*)(&buf);
                recv_packet->t4 = Time::GetCurrTimeStampUs();
                if (recv_packet->seq != _seq) {
                    ADF_EARLY_LOG << "Invalid seq " << recv_packet->seq << ", " << _seq;
                    ++_seq;
                    sleep(1);
                    continue;
                }

                int64_t delta_us = (recv_packet->t4 - recv_packet->t1  - recv_packet->t3 + recv_packet->t2) / 2;
                Time::SetTime(delta_us + recv_packet->t3);

                ADF_EARLY_LOG << "Time diff " << delta_us << "(us).";
                ++_seq;
            }
            sleep(1);
        }
    }

private:
    int _socket_fd = 0;
    std::shared_ptr<std::thread> _thread;
    bool _running = true;
    uint32_t _seq = 0;
    struct sockaddr_in _ser_addr;
};

}
}
}

using namespace hozon::netaos::tsync;

int main(int argc, char* argv[]) {
    hozon::netaos::adf_lite::SigHandler::GetInstance().Init();

    UdpTsyncSlave slave;
    slave.Init("192.168.33.42");
    // slave.Init("10.6.75.188");
    ADF_EARLY_LOG << "Succ to init slave.";

    hozon::netaos::adf_lite::SigHandler::GetInstance().NeedStopBlocking();

    return 0;
}