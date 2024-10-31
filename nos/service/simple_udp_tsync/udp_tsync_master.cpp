#include <cstdint>
#include <sys/types.h>
#include <sys/socket.h>
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

class UdpTsyncMaster {
public:
    int32_t Init() {
        _socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
        if (_socket_fd < 0) {
            ADF_EARLY_LOG << "Fail to create socket.";
            return -1;
        }

        struct sockaddr_in ser_addr;
        memset(&ser_addr, 0, sizeof(ser_addr));
        ser_addr.sin_family = AF_INET;
        ser_addr.sin_addr.s_addr = htonl(INADDR_ANY);
        ser_addr.sin_port = htons(MASTER_PORT);

        int32_t ret = bind(_socket_fd, (struct sockaddr*)&ser_addr, sizeof(ser_addr));
        if (ret < 0) {
            ADF_EARLY_LOG << "Fail to bind socket.";
            return -1;
        }

        _thread = std::make_shared<std::thread>(&UdpTsyncMaster::Routine, this);

        return 0;
    }

    ~UdpTsyncMaster() {
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
        socklen_t len;
        int32_t count;
        struct sockaddr_in client_addr;

        fd_set rfds;
        while (_running) {
            memset(buf, 0, BUFF_LEN);
            len = sizeof(client_addr);

            timeval timeout;
            timeout.tv_sec = 5;
            timeout.tv_usec = 0;
            FD_ZERO(&rfds);
            FD_SET(_socket_fd, &rfds);
            int ret = select(_socket_fd + 1, &rfds, NULL, NULL, &timeout);
            if (ret == -1) {
                ADF_EARLY_LOG << "Fail to listen on interface, " << errno;
                continue;
            } 
            else if (ret == 0) {
                ADF_EARLY_LOG << "Fail to recv";
                continue;
            } 
            else {
                count = recvfrom(_socket_fd, buf, BUFF_LEN, 0, (struct sockaddr*)&client_addr, &len);
                if(count == -1) {
                    ADF_EARLY_LOG << "Fail to recv data.";
                    continue;
                }

                if (count != sizeof(UdpTsyncPacket)) {
                    ADF_EARLY_LOG << "Invalid size " << len;
                    continue;
                }

                UdpTsyncPacket* packet = (UdpTsyncPacket*)(&buf);
                packet->t2 = Time::GetCurrTimeStampUs();
                ADF_EARLY_LOG << "Recv seq " << packet->seq;
                packet->t3 = Time::GetCurrTimeStampUs();
                ret = sendto(_socket_fd, buf, sizeof(UdpTsyncPacket), 0, (struct sockaddr*)&client_addr, len);
                ADF_EARLY_LOG << "send ret " << ret;
            }
        }
    }

private:
    int _socket_fd = 0;
    std::shared_ptr<std::thread> _thread;
    bool _running = true;
};

}
}
}


using namespace hozon::netaos::tsync;

int main(int argc, char* argv[]) {
    hozon::netaos::adf_lite::SigHandler::GetInstance().Init();

    UdpTsyncMaster master;
    master.Init();
    ADF_EARLY_LOG << "Succ to init master.";

    hozon::netaos::adf_lite::SigHandler::GetInstance().NeedStopBlocking();

    return 0;
}