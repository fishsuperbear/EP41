/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: devm server socket os
 */

#ifndef DEVM_SERVER_SOCKET_OS_H_
#define DEVM_SERVER_SOCKET_OS_H_

#include <sys/un.h>
#include <netinet/in.h>
#include <stdint.h>


namespace hozon {
namespace netaos {
namespace devm_server {

class DevmSocketOS {
 public:
    static int32_t SetCloexecOrClose(int32_t fd);
    static int32_t CreateSocket(int32_t domain, int32_t type, int32_t protocol);
    static int32_t Accept(int32_t sockfd, struct sockaddr *addr, socklen_t *addrlen);
    static int32_t Connect(int32_t sockfd, struct sockaddr *addr, socklen_t addrlen);
};


}  // namespace devm_server
}  // namespace netaos
}  // namespace hozon
#endif  // DEVM_SERVER_SOCKET_OS_H_
/* EOF */
