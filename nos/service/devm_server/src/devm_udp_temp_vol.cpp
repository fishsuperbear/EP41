/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: devm socket
 */
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <thread>


#include <iostream>
#include "devm_socket_os.h"
#include "devm_server_logger.h"
#include "devm_udp_temp_vol.h"
#include "devm_data_gathering.h"
#include "function_statistics.h"

namespace hozon {
namespace netaos {
namespace devm_server {


DevmUdpTempAndVol::DevmUdpTempAndVol() {
    stop_flag_ = false;
    ifname_ = "";
#if defined(BUILD_FOR_ORIN)
    ifname_ = "mgbe3_0.90";
#endif
    port_ = 23459;
}

DevmUdpTempAndVol::~DevmUdpTempAndVol() {
}

int32_t
DevmUdpTempAndVol::WriteVersionToCfg()
{
    return 0;
}

int32_t
DevmUdpTempAndVol::GetIp(const char *ifname, char *ip, int32_t iplen) {
    DEVM_LOG_INFO << "<DoipNetlink> get ip by ifname: " << ifname;
    if (NULL == ifname || NULL == ip) {
        return -1;
    }

    int32_t fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        return -1;
    }

    struct sockaddr_in sin;
    struct ifreq ifr;
    memset(&ifr, 0, sizeof ifr);
    strncpy(ifr.ifr_name, ifname, IFNAMSIZ - 1);
    ifr.ifr_name[IFNAMSIZ - 1] = 0;

    if (ioctl(fd, SIOCGIFADDR, &ifr) < 0) {
        DEVM_LOG_ERROR << "<DoipNetlink> call ioctl is failed! err_message: " << strerror(errno);
        close(fd);
        return -1;
    }

    memcpy(&sin, &ifr.ifr_addr, sizeof sin);
    snprintf(ip, iplen, "%s", inet_ntoa(sin.sin_addr));

    close(fd);
    return 0;
}

void
DevmUdpTempAndVol::Run() {

    while (!stop_flag_) {
        SetFd(DevmSocketOS::CreateSocket(AF_INET, SOCK_DGRAM, 0));
        if (GetFd() == -1) {
            DEVM_LOG_ERROR << "CreateSocket error, " << strerror(errno);
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;
        }
        char ip[100] = {0};
        GetIp(ifname_.c_str(), ip, 100);

        struct sockaddr_in localAddress;
        localAddress.sin_family = AF_INET;
        localAddress.sin_port = htons(port_);  // 本地端口号
        struct in_addr in_addr;
        if (inet_pton(AF_INET, ip, &in_addr) > 0) {
            localAddress.sin_addr.s_addr = inet_addr(ip);//INADDR_ANY;
        }
        else {
            localAddress.sin_addr.s_addr = INADDR_ANY;
        }

        if (bind(GetFd(), (struct sockaddr*)&localAddress, sizeof(localAddress)) == -1) {
            DEVM_LOG_ERROR << "UDP bind error, " << strerror(errno);
            CloseFd();
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;
        }
        DEVM_LOG_INFO << "UDP listen ip " << ip << ", port " << port_;

        uint8_t buffer[200];
        socklen_t add_size = sizeof(addr_);
        while (!stop_flag_) {
            fd_set rfds;
            struct timeval tv;
            int retval;

            FD_ZERO(&rfds);
            FD_SET(GetFd(), &rfds);
            tv.tv_sec = 1;
            tv.tv_usec = 0;

            retval = select(GetFd() + 1, &rfds, NULL, NULL, &tv);
            if (retval == -1) {
                DEVM_LOG_ERROR << "select error, " << strerror(errno);
                break;
            }
            else if (retval == 0) {
                DEVM_LOG_TRACE << "UDP Timeout reached.";
                continue;
            }
            else {
                memset(buffer, 0, sizeof(buffer));
                ssize_t bytesRead = recvfrom(GetFd(), buffer, sizeof(buffer), 0, (struct sockaddr*)&addr_, &add_size);
                if (bytesRead < 0) {
                    DEVM_LOG_ERROR << "recv data error, " << strerror(errno);
                    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                    break;
                }
                else if (bytesRead == 0) {
                    DEVM_LOG_ERROR << "udp server close.";
                    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                    break;
                }
                else {
                    DEVM_LOG_TRACE << "bytesRead " << bytesRead << " recv udp from " << inet_ntoa(addr_.sin_addr) << ":" << ntohs(addr_.sin_port);
                    TemperatureData temp{};
                    VoltageData vol{};
                    if (bytesRead >= 16 + 2 + 4) {
                        memcpy(&temp, buffer, 16);
                        memcpy(&vol.kl15_vol, buffer + 16, 2);
                        memcpy(&vol.kl30_vol, buffer + 16 + 2, 4);
                    }
                    if (temp.soc_temp != temperature_.soc_temp
                        || temp.mcu_temp != temperature_.mcu_temp
                        || temp.ext0_temp != temperature_.ext0_temp
                        || temp.ext1_temp != temperature_.ext1_temp
                        || vol.kl15_vol != voltage_.kl15_vol
                        || vol.kl30_vol != voltage_.kl30_vol) {
                        temperature_ = temp;
                        voltage_ = vol;
                        ;//写cfg
                    }
                    DEVM_LOG_TRACE << "soc_temp: " << temp.soc_temp;
                    DEVM_LOG_TRACE << "mcu_temp: " << temp.mcu_temp;
                    DEVM_LOG_TRACE << "ext0_temp: " << temp.ext0_temp;
                    DEVM_LOG_TRACE << "ext1_temp: " << temp.ext1_temp;
                    DEVM_LOG_TRACE << "kl15_vol: " << vol.kl15_vol;
                    DEVM_LOG_TRACE << "kl30_vol: " << vol.kl30_vol;
                    TemperatureDataInfo::getInstance()->SetTemperature(temp);
                    VoltageDataInfo::getInstance()->SetVoltage(vol);
                }
            }
        }
        CloseFd();
        DEVM_LOG_INFO << "DevmUdpTempAndVol thread exited ";

    }
}
void
DevmUdpTempAndVol::SetStopFlag() {
    stop_flag_ = true;
}

}  // namespace devm_server
}  // namespace netaos
}  // namespace hozon
/* EOF */
