/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip connection
 */
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "diag/doip/include/base/doip_connection.h"


namespace hozon {
namespace netaos {
namespace diag {


DoipConnection::DoipConnection(int32_t fd) : fd_(fd) {
}

DoipConnection::~DoipConnection() {
    shutdown(fd_, SHUT_RDWR);
    close(fd_);
    fd_ = -1;
}

int32_t
DoipConnection::GetFd() {
    return fd_;
}

void
DoipConnection::SetFd(int32_t fd) {
    fd_ = fd;
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
/* EOF */
