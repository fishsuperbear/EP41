/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip socket
 */
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "diag/doip/include/socket/doip_socket.h"

namespace hozon {
namespace netaos {
namespace diag {


DoipSocket::DoipSocket() : fd_(-1), type_(0) {
}

DoipSocket::~DoipSocket() {
}

DoipIpv4Socket::DoipIpv4Socket() {
}

DoipIpv4Socket::~DoipIpv4Socket() {
}


DoipIpv6Socket::DoipIpv6Socket() {
}

DoipIpv6Socket::~DoipIpv6Socket() {
}



}  // namespace diag
}  // namespace netaos
}  // namespace hozon
/* EOF */
