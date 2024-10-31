/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip socket os
 */

#ifndef MIDDLEWARE_DIAG_DOIP_INCLUDE_SOCKET_DOIP_SOCKET_OS_H_
#define MIDDLEWARE_DIAG_DOIP_INCLUDE_SOCKET_DOIP_SOCKET_OS_H_

#include <sys/un.h>
#include <netinet/in.h>
#include <stdint.h>

#include "diag/doip/include/socket/doip_socket.h"

namespace hozon {
namespace netaos {
namespace diag {

class DoipSocketOS {
 public:
    static int32_t SetCloexecOrClose(int32_t fd);
    static int32_t CreateSocket(int32_t domain, int32_t type, int32_t protocol);
    static int32_t Accept(int32_t sockfd, struct sockaddr *addr, socklen_t *addrlen);
    static int32_t Connect(int32_t sockfd, struct sockaddr *addr, socklen_t addrlen);
};


}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_DIAG_DOIP_INCLUDE_SOCKET_DOIP_SOCKET_OS_H_
/* EOF */
