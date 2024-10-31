/*
 * Copyright (c) Hozon SOC Co., Ltd. 2023-2023. All rights reserved.
 *
 * Description: tcp socket
 */
#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <sys/select.h>
#include <linux/rtnetlink.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <net/if.h>
#include <netdb.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>


#include "socket.h"
#include "common.h"




SocketApi::SocketApi()
{
    socket_ifd_ = -1;
    tcp_link_ = false;
}

int32_t
SocketApi::Connect(int32_t sockfd, struct sockaddr *addr, socklen_t addrlen)
{
    if (sockfd < 0 || addr == NULL) {
        return -1;
    }

    int32_t flags = fcntl(sockfd, F_SETFL, O_NONBLOCK);
    if (flags < 0) {
        FAIL_LOG << "<DoipSocketOS> os_connect set nonblock fail! fd: " << sockfd;
        return -1;
    }

    int32_t fail = 0;

    do {
        int32_t ret = connect(sockfd, addr, addrlen);
        if (ret == 0) {
            break;
        }

        if (errno != EINPROGRESS) {
            FAIL_LOG << "<DoipSocketOS> os_connect connect error!";
            fail = 1;
            break;
        }

        errno = 0;
        fd_set wset;
        FD_ZERO(&wset);
        FD_SET(sockfd, &wset);
        DEBUG_LOG << "<DoipSocketOS> os_connect fd_set size:[" << sizeof wset << "] bytes.";

        struct timeval timeout;
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;

        ret = select(sockfd + 1, NULL, &wset, NULL, &timeout);

        if (ret < 0) {
            FAIL_LOG << "<DoipSocketOS> os_connect select error!";
            fail = 1;
            break;
        }

        if (ret == 0) {
            FAIL_LOG << "<DoipSocketOS> os_connect select timeout!";
            errno = ETIMEDOUT;
            fail = 1;
            break;
        }

        if (!FD_ISSET(sockfd, &wset)) {
            FAIL_LOG << "<DoipSocketOS> os_connect FD_ISSET unknown event!";
            fail = 1;
            break;
        }

        int32_t error = -1;
        socklen_t len = sizeof(int32_t);
        if (getsockopt(sockfd, SOL_SOCKET, SO_ERROR, &error, &len) < 0) {
            FAIL_LOG << "<DoipSocketOS> os_connect getsockopt error code: " << errno << ", message: " << strerror(errno);
            fail = 1;
            break;
        }

        if (error) {
            FAIL_LOG << "<DoipSocketOS> os_connect SO_ERROR exists! error: " << error;
            errno = error;
            fail = 1;
            break;
        }
    } while (0);

    if (fcntl(sockfd, F_SETFL, flags) < 0) {
        FAIL_LOG << "<DoipSocketOS> os_connect restore fail! fd: " << sockfd;
        return -1;
    }

    if (fail) {
        return -1;
    }

    return 0;
}

int32_t
SocketApi::CreateSocket(int32_t domain, int32_t type, int32_t protocol)
{
    int32_t ifd = socket(domain, type | SOCK_CLOEXEC, protocol);
    if (ifd < 0) {
        FAIL_LOG << "socket create error." << ifd;
        return -1;
    }

    DEBUG_LOG << "socket create succ." << ifd;
    socket_ifd_ = ifd;

    return ifd;
}


#include <sys/ioctl.h>
#include <linux/can.h>
#include <linux/can/raw.h>
int32_t
SocketApi::SocketCanConfig(const char *ifName, bool noblock)
{
    int32_t ifd = socket_ifd_;
    int32_t ret;
    struct ifreq ifr;
    struct sockaddr_can addr;
    
    // Specify the network interface name (e.g. can0)
    strcpy(ifr.ifr_name, ifName);
    
    // Get the interface index
    ioctl(ifd, SIOCGIFINDEX, &ifr);

    // 设置socket为非阻塞模式
    int flags = fcntl(ifd, F_GETFL, 0);
    if (noblock) {
        fcntl(ifd, F_SETFL, flags | O_NONBLOCK);
    }
    
    // Bind the socket to the network interface
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;
    ret = bind(ifd, (struct sockaddr *)&addr, sizeof(addr));
    if (ret < 0) {
        FAIL_LOG << "socket can bind error.";
        close(ifd);
        socket_ifd_ = -1;
        return -1;
    }
    
    return 0;
}
int32_t
SocketApi::SocketCanSend(uint32_t canid, uint8_t *data, int32_t size)
{
    int32_t ifd = socket_ifd_;
    struct can_frame frame;
    int32_t ret;

    // Set the CAN ID and data for a CAN frame
    frame.can_id = canid;
    frame.can_dlc = (size <= CAN_MAX_DLEN ? size : CAN_MAX_DLEN);
    memcpy(frame.data, data, (size <= CAN_MAX_DLEN ? size : CAN_MAX_DLEN));

    // Send the CAN frame using the socket
    // uint8_t *p = (uint8_t *)&frame;
    // for (uint32_t i = 0; i < sizeof(frame); i++) {
    //     printf("%02x ", p[i]);
    // }
    // printf("\n");
    ret = write(ifd, &frame, sizeof(frame));
    if (ret != sizeof(frame)) {
        FAIL_LOG << "socket can write error " << errno << ", ret " << ret;
        close(ifd);
        socket_ifd_ = -1;
        return -1;
    }

    return ret;
}
int32_t
SocketApi::SocketCanRead(uint32_t *canid, uint8_t *data, int32_t size)
{
    int32_t ifd = socket_ifd_;
    struct can_frame frame;
    int32_t ret;

    // Receive any CAN frames
    ret = read(ifd, &frame, sizeof(frame));
    if (ret < 0) {
        FAIL_LOG << "socket can read error " << ret;
        close(ifd);
        socket_ifd_ = -1;
        return -1;
    }
    
    *canid = frame.can_id;
    memcpy(data, frame.data, (size <= frame.can_dlc ? size : frame.can_dlc));
    
    return ret;
}
int32_t
SocketApi::SocketCanRecv(uint32_t *canid, uint8_t *data, int32_t size, uint32_t timeoutms)
{
    int32_t ifd = socket_ifd_;
    struct can_frame frame;
    int32_t ret = -1;
    fd_set rset;
    
    FD_ZERO(&rset);
    FD_SET(ifd, &rset);
    do {
        struct timeval timeout = {timeoutms / 1000, (timeoutms % 1000) * 1000};
        struct timeval timenow, timeold;
        gettimeofday(&timenow, NULL);
        timeold = timenow;
        ret = select(ifd + 1, &rset, NULL, NULL, &timeout);
        if (ret == 0) {
            //timeout
            INFO_LOG << "timeout, No CAN data received";
            break;
        }
        else if (ret > 0) {
            if (FD_ISSET(ifd, &rset)) {
                ret = recvfrom(ifd, &frame, sizeof(frame), 0, NULL, NULL);
                if (ret < 0) {
                    perror("recvfrom failed");
                    close(ifd);
                    socket_ifd_ = -1;
                    break;
                }
                *canid = frame.can_id;
                memcpy(data, frame.data, (size <= frame.can_dlc ? size : frame.can_dlc));
            }
            else {
                gettimeofday(&timenow, NULL);
                timeoutms = (timenow.tv_usec > timeold.tv_usec ? timenow.tv_usec - timeold.tv_usec : timenow.tv_usec + 1000000 - timeold.tv_usec) / 1000;
                timenow.tv_sec = timenow.tv_usec > timeold.tv_usec ? timenow.tv_sec : timenow.tv_sec - 1;
                timeoutms += ((timenow.tv_sec > timeold.tv_sec ? timenow.tv_sec - timeold.tv_sec : 0) * 1000);
                continue;
            }
        }
        else {
            if (errno == EINTR) {
                gettimeofday(&timenow, NULL);
                timeoutms = (timenow.tv_usec > timeold.tv_usec ? timenow.tv_usec - timeold.tv_usec : timenow.tv_usec + 1000000 - timeold.tv_usec) / 1000;
                timenow.tv_sec = timenow.tv_usec > timeold.tv_usec ? timenow.tv_sec : timenow.tv_sec - 1;
                timeoutms += ((timenow.tv_sec > timeold.tv_sec ? timenow.tv_sec - timeold.tv_sec : 0) * 1000);
                continue;
            }
            FAIL_LOG << "socket can recv error " << errno;
            close(ifd);
            socket_ifd_ = -1;
        }
    } while(0);
    
    return ret;
}



int32_t
SocketApi::Ipv4TcpConnect(const char *ip, uint16_t port, const char *ifName)
{
{
    struct sockaddr_in addr;
    int32_t rets;
    int32_t ifd;
    
    DEBUG_LOG << "connect ip " << ip << " port " << port;
    memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = inet_addr(ip);
    
    ifd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    for (uint32_t i = 0; i < 30; i++) {

        // struct ifreq ifr;
        // memset(&ifr, 0, sizeof(ifr));
        // errno = 0;
        // strncpy(ifr.ifr_name, ifName, sizeof(ifName) - 1);
        // if (ifName == "" || setsockopt(ifd, SOL_SOCKET, SO_BINDTODEVICE, (char *)&ifr, sizeof(ifr))) {
        //     close(ifd);
        //     ifd = -1;
        //     FAIL_LOG << "<DoIPSocketHandler> Ipv4TcpCreate setsockopt SO_BINDTODEVICE code:" << errno << ", message:" << strerror(errno) << ", ifname:" << ifName;
        //     break;
        // }

        rets = connect(ifd, (struct sockaddr *)&addr, sizeof addr);
        if (rets == 0) {
            DEBUG_LOG << "connect succ.";
            break;
        }
        
        if (errno != EINPROGRESS) {
            FAIL_LOG << "<DoipSocketOS> os_connect connect error!" << "message: " << strerror(errno);
            close(ifd);
            ifd = -1;
            break;
        }
        usleep(200);
    }
    
    if (rets >= 0) {
        tcp_link_ = true;
    }
    socket_ifd_ = ifd;
    return socket_ifd_;
}


    DEBUG_LOG << "<DoIPSocketHandler> Ipv4TcpCreate ip is " << ip << ", port is " << port;
    if (ip == nullptr) {
        FAIL_LOG << "<DoIPSocketHandler> Ipv4TcpCreate ip == nullptr!";
        return -1;
    }
    socket_ifd_ = CreateSocket(AF_INET, SOCK_STREAM, 0);
    if (socket_ifd_ < 0) {
        FAIL_LOG << "<DoIPSocketHandler> CreateSocket error!";
        return -1;
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = inet_addr(ip);

    errno = 0;
    const int32_t opt = 1;
    if (setsockopt(socket_ifd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof opt)) {
        close(socket_ifd_);
        socket_ifd_ = -1;
        FAIL_LOG << "<DoIPSocketHandler> Ipv4TcpCreate setsockopt SO_REUSEADDR code:" << errno << ", message:" << strerror(errno);
        return socket_ifd_;
    }

    struct ifreq ifr;
    memset(&ifr, 0, sizeof(ifr));
    errno = 0;
    strncpy(ifr.ifr_name, ifName, sizeof(ifName) - 1);
    // if (ifName == "" || setsockopt(socket_ifd_, SOL_SOCKET, SO_BINDTODEVICE, (char *)&ifr, sizeof(ifr))) {
    //     close(socket_ifd_);
    //     socket_ifd_ = -1;
    //     FAIL_LOG << "<DoIPSocketHandler> Ipv4TcpCreate setsockopt SO_BINDTODEVICE code:" << errno << ", message:" << strerror(errno) << ", ifname:" << ifName;
    //     return socket_ifd_;
    // }
    DEBUG_LOG << "<DoIPSocketHandler> Ipv4TcpCreate bind to if: " << ifName;

    int32_t ret = Connect(socket_ifd_, (struct sockaddr *) &addr, sizeof addr);
    if (ret < 0) {
        FAIL_LOG << "<DoIPSocketHandler> Ipv4TcpCreate connect error code:" << errno << ", message:" << strerror(errno);
        close(socket_ifd_);
        socket_ifd_ = -1;
    }

    return socket_ifd_;
}

int32_t
SocketApi::Ipv4TcpSendData(uint8_t *data, int32_t dataLength)
{
    errno = 0;
    ssize_t num = send(socket_ifd_, data, dataLength, 0);
    if (num != dataLength) {
        FAIL_LOG << "<DoIPSocketHandler> SocketSend code:" << errno << ", message:" << strerror(errno) << ", fd:" << socket_ifd_ << "dataLength " << dataLength << "num " << num;
    }
    else {
        DEBUG_LOG << "<DoIPSocketHandler> SocketSend count:" << dataLength << ", num:" << (uint32_t)num << ", fd:" << socket_ifd_;
    }
    return dataLength;
}

int32_t
SocketApi::Ipv4TcpRecvData(uint8_t *recvBuf, int32_t length, uint32_t timeoutMs)
{
    fd_set rfds;
    struct timeval tv;
    int retval, recv_len;

    FD_ZERO(&rfds);
    FD_SET(socket_ifd_, &rfds);

    tv.tv_sec = timeoutMs / 1000;
    tv.tv_usec = timeoutMs % 1000 * 1000;

    retval = select(socket_ifd_ + 1, &rfds, NULL, NULL, &tv);
    if (retval == -1) {
        perror("select()");
        return -1;
    }
    else if (retval == 0) {
        // 超时
        FAIL_LOG << "Timeout reached";
        return -1;
    }
    else {
        // 可读事件已经就绪
        recv_len = recv(socket_ifd_, (char *)recvBuf, length, 0);
        if (recv_len < 0) {
            FAIL_LOG << "recv fail";
            perror("recv()");
            if(errno == ECONNRESET) {
                return -2;
            }
            return -1;
        }
        else if (recv_len == 0) {
            // 连接已经关闭
            DEBUG_LOG << "Connection closed by peer";
            return -2;
        }
        else {
            // 处理接收到的数据
            // ...
        }
    }

    return recv_len;
}

int32_t
SocketApi::Ipv4TcpShutDown()
{
    int32_t sockfd = socket_ifd_;
    if (sockfd < 0) {
        return -1;
    }

    shutdown(sockfd, SHUT_WR);
    fd_set readFds;
    FD_ZERO(&readFds);
    FD_SET(sockfd, &readFds);

    int32_t ret = select(sockfd + 1, &readFds, NULL, NULL, NULL);
    if (ret < 0) {
        FAIL_LOG << "shutdown select fail";
        return -1;
    }

    if (FD_ISSET(sockfd, &readFds)) {
        DEBUG_LOG << "shutdown Connection closed by peer";
    }
    else {
        FAIL_LOG << "shutdown fail";
        return -1;
    }

    return 0;
}


int32_t 
SocketApi::Ipv4TcpClose()
{
    if (socket_ifd_ >= 0) {
        close(socket_ifd_);
        socket_ifd_ = -1;
    }
    tcp_link_ = false;
    return 0;
}

bool 
SocketApi::GetLinkStatus()
{
    return tcp_link_;
}

