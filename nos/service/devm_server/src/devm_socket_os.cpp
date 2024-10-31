/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: devm server socket os
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

#include "devm_socket_os.h"
#include "devm_server_logger.h"

namespace hozon {
namespace netaos {
namespace devm_server {

#define CONNECTION_TIMEOUT_SEC      0
#define CONNECTION_TIMEOUT_USEC     100000 /* 100ms */


int32_t
DevmSocketOS::SetCloexecOrClose(int32_t fd) {
    if (fd == -1) {
        return -1;
    }

    do {
        int32_t flags = fcntl(fd, F_GETFD);
        if (flags == -1) {
            break;
        }

        if (fcntl(fd, F_SETFD, flags | FD_CLOEXEC) == -1) {
            break;
        }

        return fd;
    } while (0);

    close(fd);
    fd = -1;
    return fd;
}

int32_t
DevmSocketOS::CreateSocket(int32_t domain, int32_t type, int32_t protocol) {
    errno = 0;
    int32_t fd = socket(domain, type | SOCK_CLOEXEC, protocol);
    if (fd >= 0) {
        return fd;
    }
    if (errno != EINVAL) {
        return -1;
    }

    fd = socket(domain, type, protocol);
    return SetCloexecOrClose(fd);
}

int32_t
DevmSocketOS::Accept(int32_t sockfd, struct sockaddr *addr, socklen_t *addrlen) {
    int32_t fd = accept(sockfd, addr, addrlen);
    return SetCloexecOrClose(fd);
}

int32_t
DevmSocketOS::Connect(int32_t sockfd, struct sockaddr *addr, socklen_t addrlen) {
    if (sockfd < 0 || addr == NULL) {
        return -1;
    }

    int32_t flags = fcntl(sockfd, F_SETFL, O_NONBLOCK);
    if (flags < 0) {
        DEVM_LOG_ERROR << "<DevmSocketOS> os_connect set nonblock fail! fd: " << sockfd;
        return -1;
    }

    int32_t fail = 0;

    do {
        int32_t ret = connect(sockfd, addr, addrlen);
        if (ret == 0) {
            break;
        }

        if (errno != EINPROGRESS) {
            DEVM_LOG_ERROR << "<DevmSocketOS> os_connect connect error!";
            fail = 1;
            break;
        }

        errno = 0;
        fd_set wset;
        FD_ZERO(&wset);
        FD_SET(sockfd, &wset);
        DEVM_LOG_DEBUG << "<DevmSocketOS> os_connect fd_set size:[" << sizeof wset << "] bytes.";

        struct timeval timeout;
        timeout.tv_sec = CONNECTION_TIMEOUT_SEC;
        timeout.tv_usec = CONNECTION_TIMEOUT_USEC;

        ret = select(sockfd + 1, NULL, &wset, NULL, &timeout);

        if (ret < 0) {
            DEVM_LOG_ERROR << "<DevmSocketOS> os_connect select error!";
            fail = 1;
            break;
        }

        if (ret == 0) {
            DEVM_LOG_WARN << "<DevmSocketOS> os_connect select timeout!";
            errno = ETIMEDOUT;
            fail = 1;
            break;
        }

        if (!FD_ISSET(sockfd, &wset)) {
            DEVM_LOG_WARN << "<DevmSocketOS> os_connect FD_ISSET unknown event!";
            fail = 1;
            break;
        }

        int32_t error = -1;
        socklen_t len = sizeof(int32_t);
        if (getsockopt(sockfd, SOL_SOCKET, SO_ERROR, &error, &len) < 0) {
            DEVM_LOG_ERROR << "<DevmSocketOS> os_connect getsockopt error code: " << errno << ", message: " << strerror(errno);
            fail = 1;
            break;
        }

        if (error) {
            DEVM_LOG_WARN << "<DevmSocketOS> os_connect SO_ERROR exists! error: " << error;
            errno = error;
            fail = 1;
            break;
        }
    } while (0);

    if (fcntl(sockfd, F_SETFL, flags) < 0) {
        DEVM_LOG_ERROR << "<DevmSocketOS> os_connect restore fail! fd: " << sockfd;
        return -1;
    }

    if (fail) {
        return -1;
    }

    return 0;
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
/* EOF */
