/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip netlink
 */

#ifndef MIDDLEWARE_DIAG_DOIP_INCLUDE_BASE_DOIP_NETLINK_H_
#define MIDDLEWARE_DIAG_DOIP_INCLUDE_BASE_DOIP_NETLINK_H_

#include <sys/socket.h>
#include <stdint.h>
#include <string>

namespace hozon {
namespace netaos {
namespace diag {

class DoipNetlink {
 public:
    DoipNetlink();
    ~DoipNetlink();
    void GetIFName(char* ifname, int32_t fd, char *ip);
    int32_t GetIp(const char *ifname, char *ip, int32_t iplen);
    int32_t GetMac(const char *ifname, char *mac);
    int32_t CheckLinkAvailable(const std::string& ifname, const char* local_addr);
};


}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_DIAG_DOIP_INCLUDE_BASE_DOIP_NETLINK_H_
/* EOF */
