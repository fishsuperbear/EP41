/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip netlink
 */
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <arpa/inet.h>
#include <string.h>
#include <sys/select.h>
#include <stdio.h>

#include "diag/doip/include/base/doip_netlink.h"
#include "diag/doip/include/base/doip_logger.h"

namespace hozon {
namespace netaos {
namespace diag {

DoipNetlink::DoipNetlink() {
}

DoipNetlink::~DoipNetlink() {
}

void
DoipNetlink::GetIFName(char* ifname, int32_t fd, char *ip) {
    DOIP_INFO << "<DoipNetlink> get if name by fd: " << fd << ", ip: " << ip;
    if (NULL == ifname || NULL == ip) {
        return;
    }

    char *ip_tmp = reinterpret_cast<char*>(malloc(strlen(ip) + 1));
    if (NULL == ip_tmp) {
        return;
    }

    memcpy(ip_tmp, ip, strlen(ip) + 1);

    struct ifconf ifc;
    char buf[512];
    struct ifreq *ifr;

    ifc.ifc_buf = buf;
    ifc.ifc_len = 512;

    if (ioctl(fd, SIOCGIFCONF, &ifc) < 0) {
        DOIP_ERROR << "<DoipNetlink> call ioctl is failed! err_message: " << strerror(errno);
        free(ip_tmp);
        return;
    }

    ifr = (struct ifreq*)buf;
    int32_t i = 0;
    for (i = (ifc.ifc_len / sizeof(struct ifreq)); i > 0; i--) {
        char *ip_t = inet_ntoa(((struct sockaddr_in*)&(ifr->ifr_addr))->sin_addr);
        if (0 == strcmp(ip_tmp, ip_t)) {
            memcpy(ifname, ifr->ifr_name, strlen(ifr->ifr_name));
            break;
        }
        ifr++;
    }

    free(ip_tmp);
}

int32_t
DoipNetlink::GetIp(const char *ifname, char *ip, int32_t iplen) {
    DOIP_INFO << "<DoipNetlink> get ip by ifname: " << ifname;
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
        DOIP_ERROR << "<DoipNetlink> call ioctl is failed! err_message: " << strerror(errno);
        close(fd);
        return -1;
    }

    memcpy(&sin, &ifr.ifr_addr, sizeof sin);
    snprintf(ip, iplen, "%s", inet_ntoa(sin.sin_addr));

    close(fd);
    return 0;
}

int32_t
DoipNetlink::GetMac(const char *ifname, char *mac) {
    DOIP_INFO << "<DoipNetlink> get mac by ifname: " << ifname;
    if (NULL == ifname || NULL == mac) {
        return -1;
    }

    int32_t fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        return -1;
    }

    struct ifreq ifr;
    memset(&ifr, 0, sizeof ifr);
    strncpy(ifr.ifr_name, ifname, IFNAMSIZ - 1);
    ifr.ifr_name[IFNAMSIZ - 1] = 0;

    if (ioctl(fd, SIOCGIFHWADDR, &ifr) < 0) {
        DOIP_ERROR << "<DoipNetlink> call ioctl is failed! err_message: " << strerror(errno);
        close(fd);
        return -1;
    }

    memcpy(mac, ifr.ifr_hwaddr.sa_data, 6);
    close(fd);

    return 0;
}

int32_t
DoipNetlink::CheckLinkAvailable(const std::string& ifname, const char* local_addr) {
    if (ifname  == "") {
        return 0;
    }

    int link_status = 0;
    int ip_status = 0;

    int ioctlSfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (ioctlSfd < 0) {
        DOIP_ERROR << "<DoipNetlink> CheckLinkAvailable socket() failed! err_message: " << strerror(errno);
        return 0;
    }

    struct ifreq ifr;
    memset(&ifr, 0, sizeof(struct ifreq));
    strncpy(ifr.ifr_name, ifname.c_str(), 16);
    ifr.ifr_name[15] = 0;

    struct in_addr ip_addr;
    unsigned flags1;

    if (ioctl(ioctlSfd, SIOCGIFADDR, &ifr) < 0) {
        ip_addr.s_addr = 0;
    } else {
        ip_addr.s_addr = ((struct sockaddr_in*) &ifr.ifr_addr)->sin_addr.s_addr;
    }

    if (ioctl(ioctlSfd, SIOCGIFFLAGS, &ifr) < 0) {
        flags1 = 0;
    } else {
        flags1 = ifr.ifr_flags;
    }

    if (0 != ip_addr.s_addr && (flags1 & IFF_UP)) {
        link_status = 1;
    } else {
        link_status = 0;
    }

    char* address = inet_ntoa(ip_addr);
    if (nullptr == local_addr || strlen(local_addr) == 0) {
        if (0 != ip_addr.s_addr) {
            ip_status = 1;
        } else {
            ip_status = 0;
        }
    } else {
        if (0 == strcmp(address, local_addr)) {
            ip_status = 1;
        } else {
            ip_status = 0;
        }
    }

    close(ioctlSfd);

    return (link_status * 10 + ip_status);
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
/* EOF */
