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
#include "devm_data_define.h"
#include "devm_socket_os.h"
#include "devm_server_logger.h"
#include "devm_udp_mcu_version.h"
#include "device_info.h"
#include "devm_data_gathering.h"
#include "cfg/include/config_param.h"
#include "function_statistics.h"
#include "cfg_data.hpp"

namespace hozon {
namespace netaos {
namespace devm_server {


DevmUdpMcuVersion::DevmUdpMcuVersion() {
    stop_flag_ = false;
    ifname_ = "";
#if defined(BUILD_FOR_ORIN)
    ifname_ = "mgbe3_0.90";
#endif
    port_ = 23457;
}

DevmUdpMcuVersion::~DevmUdpMcuVersion() {
}

void
DevmUdpMcuVersion::Init() {
    FunctionStatistics func("DevmUdpMcuVersion::Init, ");
    ConfigParam::Instance()->GetParam(MCU_VERSION_DYNAMIC, mcu_version_);
    CfgResultCode res = ConfigParam::Instance()->GetParam(SWT_VERSION_DIDS, swt_version_);
    if (res != CONFIG_OK) {
        swt_version_ = CfgValueInfo::getInstance()->GetCfgValueFromFile("/cfg/version/version.json", SWT_VERSION_DIDS);
    }
    ConfigParam::Instance()->GetParam(SWT_VERSION_DYNAMIC, swt_version_dyna_);
    ConfigParam::Instance()->GetParam(USS_VERSION_DYNAMIC, uss_version_dyna_);
    DeviceInfomation::getInstance()->SetUssVersion(uss_version_dyna_);
}
void
DevmUdpMcuVersion::DeInit() {
    FunctionStatistics func("DevmUdpMcuVersion::DeInit finish, ");
}

int32_t
DevmUdpMcuVersion::WriteVersionToCfg()
{
    return 0;
}

int32_t
DevmUdpMcuVersion::GetIp(const char *ifname, char *ip, int32_t iplen) {
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
DevmUdpMcuVersion::Run() {

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

        char buffer[200]{};
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
                    DEVM_LOG_DEBUG << "bytesRead " << bytesRead << " recv udp from " << inet_ntoa(addr_.sin_addr) << ":" << ntohs(addr_.sin_port) << " message: " << buffer << " message: " << buffer+64;
                    char ver_tmp[200]{};
                    memcpy(ver_tmp, buffer, 64);
                    std::string mcu_version(ver_tmp);
                    memset(ver_tmp, 0, sizeof(ver_tmp));
                    memcpy(ver_tmp, buffer + 64, 100);
                    std::string swt_version(ver_tmp);
                    memset(ver_tmp, 0, sizeof(ver_tmp));
                    memcpy(ver_tmp, buffer + 64 + 100, 8);
                    std::string uss_version(ver_tmp);

                    DEVM_LOG_DEBUG << "mcu verison: " << mcu_version;
                    DeviceInfomation::getInstance()->SetMcuVersion(mcu_version);
                    if (mcu_version != mcu_version_) {
                        mcu_version_ = mcu_version;
                        DEVM_LOG_INFO << "write cfg dynamic mcu verison: " << mcu_version;
                        ConfigParam::Instance()->SetParam<std::string>(MCU_VERSION_DYNAMIC, mcu_version, ConfigPersistType::CONFIG_NO_PERSIST);
                    }
                    // 截取冒号后的版本，并截取\r\n前 "Switch Version: XPC_XS_88Q6113_20231026_C8088A\r\n"
                    size_t colonPos = swt_version.find("Switch Version:");
                    if (colonPos != std::string::npos) {
                        swt_version = swt_version.substr(colonPos+strlen("Switch Version:")+1);
                    }
                    colonPos = swt_version.find('\r');
                    if (colonPos != std::string::npos) {
                        swt_version = swt_version.substr(0, colonPos-1);
                    }
                    DEVM_LOG_DEBUG << "swt verison: " << swt_version;
                    DeviceInfomation::getInstance()->SetSwtVersion(swt_version);
                    if (swt_version != swt_version_) {
                        swt_version_ = swt_version;
                        DEVM_LOG_INFO << "write cfg swt verison: " << swt_version;
                        ConfigParam::Instance()->SetParam<std::string>(SWT_VERSION_DIDS, swt_version, ConfigPersistType::CONFIG_SYNC_PERSIST);
                    }
                    if (swt_version != swt_version_dyna_) {
                        swt_version_dyna_ = swt_version;
                        DEVM_LOG_INFO << "write cfg swt dynamic verison: " << swt_version;
                        DeviceInfomation::getInstance()->SetSwtVersion(swt_version);
                        ConfigParam::Instance()->SetParam<std::string>(SWT_VERSION_DYNAMIC, swt_version, ConfigPersistType::CONFIG_NO_PERSIST);
                    }

                    DEVM_LOG_DEBUG << "uss verison: " << uss_version;
                    DeviceInfomation::getInstance()->SetUssVersion(uss_version);
                    if (uss_version != uss_version_dyna_) {
                        uss_version_dyna_ = uss_version;
                        DEVM_LOG_INFO << "write cfg uss dynamic verison: " << uss_version;
                        ConfigParam::Instance()->SetParam<std::string>(USS_VERSION_DYNAMIC, uss_version, ConfigPersistType::CONFIG_NO_PERSIST);
                    }
                }
            }
        }
        CloseFd();
        DEVM_LOG_INFO << "DevmUdpMcuVersion thread exited ";
    }
}
void
DevmUdpMcuVersion::SetStopFlag() {
    stop_flag_ = true;
}

}  // namespace devm_server
}  // namespace netaos
}  // namespace hozon
/* EOF */
