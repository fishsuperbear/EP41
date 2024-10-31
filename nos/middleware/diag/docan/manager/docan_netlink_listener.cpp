/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanNetlinkListener implement
 */

#include "docan_netlink_listener.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <linux/types.h>
#include <linux/netlink.h>
#include <errno.h>
#include <sys/poll.h>

#include "diag/docan/log/docan_log.h"
#include "diag/docan/manager/docan_task_runner.h"

namespace hozon {
namespace netaos {
namespace diag {

DocanNetlinkListener* DocanNetlinkListener::s_instance = nullptr;

DocanNetlinkListener* DocanNetlinkListener::instance()
{
    if (nullptr == s_instance) {
        s_instance = new DocanNetlinkListener();
    }
    return s_instance;
}

void DocanNetlinkListener::destory()
{
    if (nullptr != s_instance) {
        delete s_instance;
        s_instance = nullptr;
    }
}

DocanNetlinkListener::DocanNetlinkListener()
{
    memset(m_buff, 0x00, sizeof(m_buff));
}

DocanNetlinkListener::~DocanNetlinkListener()
{
    m_stopFlag = true;
    if (m_sockfd > 0) {
        ::close(m_sockfd);
        m_sockfd = -1;
    }
}

int32_t DocanNetlinkListener::Init(void)
{
    int32_t ret = -1;
    ret = setupNetlinkSocket();
    return ret;
}

int32_t DocanNetlinkListener::Start(void)
{
    m_stopFlag = false;
    m_pollThread = std::thread(&DocanNetlinkListener::PollThread, this);
    return 0;
}

int32_t DocanNetlinkListener::Stop(void)
{
    m_stopFlag = true;
    if (m_pollThread.joinable()) {
        m_pollThread.join();
    }
    return 0;
}

int32_t DocanNetlinkListener::Deinit(void)
{
    m_stopFlag = true;
    if (m_sockfd > 0) {
        ::close(m_sockfd);
        m_sockfd = -1;
    }
    return 0;
}

void DocanNetlinkListener::parse(uint8_t* buff, uint32_t size)
{
    if (nullptr == buff || size == 0) {
        return;
    }
}

void DocanNetlinkListener::postEvent(DocanTaskEvent *ev)
{
    if (nullptr == ev) {
        DOCAN_LOG_E("ev is nullptr!");
        return;
    }
    DocanTaskRunner::instance()->post(ev);
}

int32_t DocanNetlinkListener::setupNetlinkSocket()
{
    int32_t ret = -1;
    if (m_sockfd > 0 ) {
        DOCAN_LOG_W("Netlink socket is existed, m_sockfd: %d", m_sockfd);
        return m_sockfd;
    }

    //create Netlink socket to receive uEvent
    int32_t sockfd = socket(PF_NETLINK, SOCK_DGRAM | SOCK_CLOEXEC,
            NETLINK_KOBJECT_UEVENT);
    if (sockfd < 0) {
        DOCAN_LOG_E("Unable to create uevent socket: %s", strerror(errno));
        return ret;
    }

    do {
        int32_t sz = DOCAN_NETLINK_BUFF_SIZE;
        ret = setsockopt(sockfd, SOL_SOCKET, SO_RCVBUFFORCE, &sz, sizeof(sz));
        if (ret < 0) {
            DOCAN_LOG_E("Unable to set uevent socket SO_RECBUFFORCE option: %s", strerror(errno));
            break;
        }

        int32_t on = 1;
        ret = setsockopt(sockfd, SOL_SOCKET, SO_PASSCRED, &on, sizeof(on));
        if (ret < 0) {
            DOCAN_LOG_E("Unable to set uevent socket SO_PASSCRED option: %s", strerror(errno));
            break;
        }

        struct sockaddr_nl nladdr;
        memset(&nladdr, 0, sizeof(nladdr));
        nladdr.nl_family = AF_NETLINK;
        nladdr.nl_pid = getpid();
        nladdr.nl_groups = 0xffffffff;
        ret = bind(sockfd, (struct sockaddr *) &nladdr, sizeof(nladdr));
        if (ret < 0) {
            DOCAN_LOG_E("Unable to bind uevent socket: %s", strerror(errno));
            break;;
        }
    } while (0);

    if (ret < 0) {
        ::close(sockfd);
        sockfd = -1;
        return ret;
    }

    DOCAN_LOG_I("setupNetlinkSocket successful canfd: %d" , sockfd);
    m_sockfd = sockfd;
    return sockfd;
}

int32_t DocanNetlinkListener::recvNetlinkMessage()
{
    // DOCAN_LOG_D("recvNetlinkMessage sockfd: %d", m_sockfd);
    int32_t ret = -1;
    if (-1 == m_sockfd) {
        return ret;
    }

    struct pollfd fds[1];
    fds[0].fd = m_sockfd;
    fds[0].events = POLLIN | POLLNVAL | POLLERR | POLLHUP;
    static const int32_t POLLIN_TIMEOUT = 1000;  // one second (in msec)
    ret = poll(fds, 1, POLLIN_TIMEOUT);
    if (ret < 0) {
        if (errno != EINTR) {
            DOCAN_LOG_E("poll(fds, 1, POLL_TIMEOUT) error: %s", strerror(errno));
        }
        return ret;
    }

    if (ret == 0) {
        // DOCAN_LOG_E("poll() timeout: %d msec",  POLLIN_TIMEOUT);
        return 0;
    }

    if ((fds[0].revents & POLLERR) || (fds[0].revents & POLLHUP) || (fds[0].revents & POLLNVAL)) {
        DOCAN_LOG_E("poll() reports POLLERR|POLLHUP|POLLNVAL error");
        ret = -1;
        return ret;
    }

    if (fds[0].revents & POLLIN) {
        memset(m_buff, 0x00, sizeof(m_buff));
        ret = recv(m_sockfd, m_buff, sizeof(m_buff), MSG_DONTWAIT);
        if (ret <= 0) {
            // DOCAN_LOG_E("recv(m_sockfd, &m_buff, sizeof(m_buff), MSG_DONTWAIT) faild, m_sockfd: %d, errno: %s", m_sockfd, strerror(errno));
            return ret;
        }
        // DOCAN_LOG_D("recv netlink msg: %s", m_buff);
        parse(m_buff, ret);
    }
    // fill into the data info

    return ret;
}

int32_t DocanNetlinkListener::PollThread()
{
    DOCAN_LOG_D("PollThread successful sockfd: %d", m_sockfd);
    int32_t ret = -1;
    if (m_sockfd < 0) {
        DOCAN_LOG_E("PollThread invalid sockfd: %d", m_sockfd);
        return ret;
    }

    ret = m_sockfd;
    while (!m_stopFlag) {
        if (recvNetlinkMessage() <= 0) {
            continue;
        }

        {
            // std::lock_guard<std::mutex> lck(recv_queue_mutex_);
            // if (recv_queue_.size() < DOCAN_QUQUE_BUFF_SIZE) {
            //     recv_queue_.push_back(packet);
            // }
            // ret = recv_queue_.size();
        }
    }
    return ret;
}


} // end of diag
} // end of netaos
} // end of hozon
/* EOF */