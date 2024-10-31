/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip select
 */

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <errno.h>

#include "diag/doip/include/base/doip_select.h"
#include "diag/doip/include/base/doip_logger.h"

namespace hozon {
namespace netaos {
namespace diag {

#define FD_SET_SIZE     1024
#define FD_SET_BITS     (8 * sizeof(fd_mask))

#define FD_BITS_TO_BYTES(n) \
    ((((n) - 1) / FD_SET_BITS + 1) * sizeof(fd_mask))


DoipSelect *DoipSelect::instancePtr_ = nullptr;
std::mutex DoipSelect::instance_mtx_;

DoipSelect *DoipSelect::Instance() {
    if (nullptr == instancePtr_) {
        std::lock_guard<std::mutex> lck(instance_mtx_);
        if (nullptr == instancePtr_) {
            instancePtr_ = new DoipSelect();
        }
    }
    return instancePtr_;
}

DoipSelect::DoipSelect() {
}

DoipSelect::~DoipSelect() {
}

void
DoipSelect::Create() {
    fd_setsize_ = FD_BITS_TO_BYTES(FD_SET_SIZE);
    event_readset_in_ = reinterpret_cast<fd_set*>(malloc(fd_setsize_));
    event_readset_out_ = reinterpret_cast<fd_set*>(malloc(fd_setsize_));
    event_writeset_in_ = reinterpret_cast<fd_set*>(malloc(fd_setsize_));
    event_writeset_out_ = reinterpret_cast<fd_set*>(malloc(fd_setsize_));

    memset(event_readset_in_, 0, fd_setsize_);
    memset(event_readset_out_, 0, fd_setsize_);
    memset(event_writeset_in_, 0, fd_setsize_);
    memset(event_writeset_out_, 0, fd_setsize_);

    max_fd_ = 0;
    notify_fd_[0] = -1;
    notify_fd_[1] = -1;

    if (InitNotify() < 0) {
        Destroy();
    }
}

void
DoipSelect::Destroy() {
    if (event_readset_in_ != NULL) {
        free(event_readset_in_);
    }
    if (event_readset_out_ != NULL) {
        free(event_readset_out_);
    }
    if (event_writeset_in_ != NULL) {
        free(event_writeset_in_);
    }
    if (event_writeset_out_ != NULL) {
        free(event_writeset_out_);
    }

    {
        std::lock_guard<std::recursive_mutex> lck(mtx_);
        event_list_.clear();
    }

    if (notify_fd_[0] > 0) {
        close(notify_fd_[0]);
    }

    if (notify_fd_[1]) {
        close(notify_fd_[1]);
    }
}

int32_t
DoipSelect::InitNotify() {
    if (InternalSocket(notify_fd_) < 0) {
        return -1;
    }

    int32_t fail = 0;
    int32_t fd0 = notify_fd_[0];
    int32_t fd1 = notify_fd_[1];

    do {
        // make two sockets nonblocking
        int32_t flags = fcntl(fd0, F_GETFL, NULL);
        if (flags < 0) {
            DOIP_DEBUG << "<DoipSelect> init_notification fd0 fcntl(" << fd0 << ", F_GETFL)";
            fail = 1;
            break;
        }
        if (fcntl(fd0, F_SETFL, flags | O_NONBLOCK) == -1) {
            DOIP_DEBUG << "<DoipSelect> init_notification fd0 fcntl(" << fd0 << ", F_SETFL)";
            fail = 1;
            break;
        }
        flags = fcntl(fd1, F_GETFL, NULL);
        if (flags < 0) {
            DOIP_DEBUG << "<DoipSelect> init_notification fd1 fcntl(" << fd1 << ", F_GETFL)";
            fail = 1;
            break;
        }
        if (fcntl(fd1, F_SETFL, flags | O_NONBLOCK) == -1) {
            DOIP_DEBUG << "<DoipSelect> init_notification fd1 fcntl(" << fd1 << ", F_SETFL)";
            fail = 1;
            break;
        }

        doip_event_t event;
        event.ev_type = DOIP_EV_INTER;
        event.data.ptr = NULL;
        event.events = DOIP_EV_READ;

        // event_list_.push_back(&event);

        if (SelectAdd(fd0, &event) < 0) {
            fail = 1;
            break;
        }
    } while (0);

    if (fail) {
        close(fd0);
        notify_fd_[0] = -1;
        close(fd1);
        notify_fd_[1] = -1;
        return -1;
    }

    return 0;
}

int32_t
DoipSelect::InternalSocket(int32_t notify_fd_[2]) {
    int32_t listener = socket(AF_INET, SOCK_STREAM, 0);
    if (listener < 0) {
        DOIP_ERROR << "<DoipSelect> InternalSocket listener fd create error!";
        return -1;
    }

    struct sockaddr_in listen_addr;
    socklen_t size = sizeof listen_addr;
    memset(&listen_addr, 0, size);
    listen_addr.sin_family = AF_INET;
    listen_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    listen_addr.sin_port = 0;

    errno = 0;
    if (bind(listener, (struct sockaddr *)&listen_addr, size) < 0) {
        close(listener);
        listener = -1;
        DOIP_ERROR << "<DoipSelect> InternalSocket bind error code: " << errno << ", message: " << strerror(errno);
        return listener;
    }

    errno = 0;
    if (listen(listener, 1) < 0) {
        close(listener);
        listener = -1;
        DOIP_ERROR << "<DoipSelect> InternalSocket listen error code: " << errno << ", message: " << strerror(errno);
        return listener;
    }


    int32_t connector = socket(AF_INET, SOCK_STREAM, 0);
    if (connector < 0) {
        close(listener);
        listener = -1;
        DOIP_ERROR << "<DoipSelect> InternalSocket connector fd create error!";
        return listener;
    }

    errno = 0;
    struct sockaddr_in connect_addr;
    size = sizeof connect_addr;
    if (getsockname(listener, (struct sockaddr *)&connect_addr, &size) < 0) {
        close(listener);
        close(connector);
        listener = -1;
        connector = -1;
        DOIP_ERROR << "<DoipSelect> InternalSocket getsockname error code: " << errno << ", message: " << strerror(errno);
        return -1;
    }
    char *ip = inet_ntoa(connect_addr.sin_addr);
    int32_t port = ntohs(connect_addr.sin_port);
    DOIP_INFO << "<DoipSelect> InternalSocket connector_fd: " << connector << ", addr: " << ip << ", port: " << port;

    errno = 0;
    if (connect(connector, (struct sockaddr *)&connect_addr, size) < 0) {
        close(listener);
        close(connector);
        listener = -1;
        connector = -1;
        DOIP_ERROR << "<DoipSelect> InternalSocket connect error code: " << errno << ", message: " << strerror(errno);
        return -1;
    }

    errno = 0;
    size = sizeof(listen_addr);
    int32_t acceptor = accept(listener, (struct sockaddr *)&listen_addr, &size);
    if (acceptor < 0) {
        close(listener);
        close(connector);
        listener = -1;
        connector = -1;
        DOIP_ERROR << "<DoipSelect> InternalSocket accept error code: " << errno << ", message: " << strerror(errno);
        return -1;
    }
    ip = inet_ntoa(listen_addr.sin_addr);
    port = ntohs(listen_addr.sin_port);
    DOIP_INFO << "<DoipSelect> InternalSocket acceptor_fd: " << acceptor << ", addr: " << ip << ", port: " << port;

    close(listener);
    notify_fd_[0] = connector;
    notify_fd_[1] = acceptor;

    return 0;
}

int32_t
DoipSelect::Control(int32_t type, int32_t fd, doip_event_t *event) {
    int32_t ret = -1;

    switch (type) {
    case DOIP_SEL_ADD:
        ret = SelectAdd(fd, event);
        break;
    case DOIP_SEL_MOD:
        ret = SelectDel(fd);
        if (ret == 0) {
            ret = SelectAdd(fd, event);
        }
        break;
    case DOIP_SEL_DEL:
        ret = SelectDel(fd);
        break;
    default:
        break;
    }

    return ret;
}

int32_t
DoipSelect::SelectAdd(int32_t fd, doip_event_t *event) {
    std::lock_guard<std::recursive_mutex> lck(mtx_);

    doip_event_t *ev = SelectFind(fd);
    if (ev != NULL) {
        DOIP_WARN << "<DoipSelect> select add fd: " << fd << " has existed!";
        return -1;
    }

    ev = reinterpret_cast<doip_event*>(malloc(sizeof *ev));
    *ev = *event;
    ev->fd = fd;
    event_list_.push_back(ev);
    DOIP_INFO << "<DoipSelect> select add fd: " << fd;

    if (max_fd_ < fd) {
        max_fd_ = fd;
    }

    int32_t events = event->events;
    if (events & DOIP_EV_READ) {
        FD_SET(fd, event_readset_in_);
    }

    if (events & DOIP_EV_WRITE) {
        FD_SET(fd, event_writeset_in_);
    }

    int32_t ret = Notify();
    if (ret < 0) {
        DOIP_WARN << "<DoipSelect> failed to notify!";
    }

    return 0;
}

int32_t
DoipSelect::SelectDel(int32_t fd) {
    std::lock_guard<std::recursive_mutex> lck(mtx_);

    doip_event_t *ev = SelectFind(fd);
    if (ev == NULL) {
        DOIP_WARN << "<DoipSelect> SelectDel can not find fd: " << fd;
        return -1;
    }

    int32_t events = ev->events;

    event_list_.remove(ev);
    free(ev);

    if (events & DOIP_EV_READ) {
        FD_CLR(fd, event_readset_in_);
    }

    if (events & DOIP_EV_WRITE) {
        FD_CLR(fd, event_writeset_in_);
    }

    int32_t ret = Notify();
    if (ret < 0) {
        DOIP_WARN << "<DoipSelect> failed to notify!";
    }

    return 0;
}

int32_t
DoipSelect::Notify() {
    std::lock_guard<std::recursive_mutex> lck(mtx_);

    if (notify_fd_[1] < 0) {
        return -1;
    }

    char buf[1];
    buf[0] = static_cast<char>(0);
    errno = 0;
    ssize_t r = write(notify_fd_[1], buf, 1);
    if (r <= 0) {
        DOIP_ERROR << "<DoipSelect> Notify() write error code: " << errno << ", message: " << strerror(errno);
        return -1;
    }

    return ((r < 0) && (errno != EAGAIN)) ? -1 : 0;
}

int32_t
DoipSelect::Dispatch(doip_event_t *event, int32_t length, int32_t timeout) {
    struct timeval *tv = NULL;
    struct timeval temp;
    if (timeout >= 0) {
        temp.tv_sec = timeout / 1000;
        temp.tv_usec = (timeout % 1000) * 1000;
        tv = &temp;
    }

    {
        std::lock_guard<std::recursive_mutex> lck(mtx_);
        memcpy(event_readset_out_, event_readset_in_, fd_setsize_);
        memcpy(event_writeset_out_, event_writeset_in_, fd_setsize_);
    }

    errno = 0;
    int32_t count = 0;
    int32_t nfds = max_fd_ + 1;
    int32_t res = select(nfds, event_readset_out_, event_writeset_out_, NULL, tv);

    std::lock_guard<std::recursive_mutex> lck(mtx_);
    if (res < 0) {
        if (errno != EINTR) {
            DOIP_ERROR << "<DoipSelect> dispach error!";
            return -1;
        }

        return 0;
    }

    for (int32_t i = 0; i < nfds; ++i) {
        res = 0;
        if (FD_ISSET(i, event_readset_out_)) {
            res |= DOIP_EV_READ;
        }
        if (FD_ISSET(i, event_writeset_out_)) {
            res |= DOIP_EV_WRITE;
        }

        if (res == 0) {
            continue;
        }

        if (count >= length) {
            DOIP_WARN << "<DoipSelect> event queue is full[i:" << i << "][length:" << length << "]";
            break;
        }

        doip_event_t *ev = SelectFind(i);
        if (ev == NULL) {
            DOIP_WARN << "<DoipSelect> Dispatch can not find fd: " << i;
            continue;
        } else {
            if (ev->ev_type == DOIP_EV_INTER) {
                SelectResponse();
                continue;
            }

            event[count] = *ev;
            event[count].events = res;
            ++count;
        }
    }

    return count;
}

int32_t
DoipSelect::SelectResponse() {
    char buf[1] = {0};
    errno = 0;
    ssize_t r = read(notify_fd_[0], buf, 1);
    if (r <= 0) {
        DOIP_ERROR << "<DoipSelect> SelectResponse read failed! code: " << errno << ", message: " << strerror(errno);
        return -1;
    }

    return 0;
}

doip_event_t *
DoipSelect::SelectFind(int32_t fd) {
    if (fd < 0) {
        DOIP_ERROR << "<DoipSelect> select find fd < 0 !";
        return NULL;
    }

    std::list<doip_event_t*>::iterator it;
    for (it = event_list_.begin(); it != event_list_.end(); ++it) {
        if ((*it)->fd == fd) {
            return *it;
        }
    }

    return NULL;
}


}  // namespace diag
}  // namespace netaos
}  // namespace hozon
/* EOF */
