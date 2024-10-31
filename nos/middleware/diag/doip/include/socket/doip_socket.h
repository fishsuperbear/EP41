/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip socket
 */

#ifndef MIDDLEWARE_DIAG_DOIP_INCLUDE_SOCKET_DOIP_SOCKET_H_
#define MIDDLEWARE_DIAG_DOIP_INCLUDE_SOCKET_DOIP_SOCKET_H_

#include <sys/un.h>
#include <netinet/in.h>
#include <stdint.h>

#include "diag/doip/include/base/doip_event_loop.h"

namespace hozon {
namespace netaos {
namespace diag {


class DoipSocket {
 public:
    DoipSocket();
    ~DoipSocket();
    int32_t GetFd() {return fd_;}
    void SetFd(int32_t fd) {fd_ = fd;}
    void CloseFd() {close(fd_);}
 public:
    int32_t fd_;
    int32_t type_;
    doip_event_source_t* source;
};

class DoipIpv4Socket : public DoipSocket {
 public:
    DoipIpv4Socket();
    ~DoipIpv4Socket();
 public:
    struct sockaddr_in addr;
};

class DoipIpv6Socket : public DoipSocket {
 public:
    DoipIpv6Socket();
    ~DoipIpv6Socket();
 public:
    struct sockaddr_in6 addr;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_DIAG_DOIP_INCLUDE_SOCKET_DOIP_SOCKET_H_
/* EOF */
