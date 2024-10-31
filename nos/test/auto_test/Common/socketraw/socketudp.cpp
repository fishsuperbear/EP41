/*
 * Copyright (c) Hozon SOC Co., Ltd. 2023-2023. All rights reserved.
 *
 * Description: udp socket
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


#include "socketudp.h"
#include "common.h"




SocketApiUdp::SocketApiUdp()
{
    socket_ifd_ = -1;
    tcp_link_ = false;
}


int32_t
SocketApiUdp::CreateSocket(int32_t domain, int32_t type, int32_t protocol)
{
    int32_t ifd = socket(domain, type | SOCK_CLOEXEC, protocol);
    if (ifd < 0) {
        FAIL_LOG << "socket create error." << ifd;
        return -1;
    }

    DEBUG_LOG << "socket create succ, fd=" << ifd;
    socket_ifd_ = ifd;

    return ifd;
}

int32_t
SocketApiUdp::Ipv4UdpSendData(uint8_t *data, int32_t dataLength)
{
    errno = 0;
    DEBUG_LOG << "fd " << socket_ifd_ <<" ip " << inet_ntoa(udp_addr_.sin_addr) << " port " << ntohs(udp_addr_.sin_port);
    ssize_t num = sendto(socket_ifd_, data, dataLength, 0, (sockaddr*)&udp_addr_, sizeof(sockaddr_in));
    if (num != dataLength) {
        FAIL_LOG << "<DoIPSocketHandler> UDP SocketSendto code:" << errno << ", message:" << strerror(errno) << ", fd:" << socket_ifd_ << "dataLength " << dataLength << "num " << num;
    }
    else {
        DEBUG_LOG << "<DoIPSocketHandler> UDP SocketSendto count:" << dataLength << ", num:" << (uint32_t)num << ", fd:" << socket_ifd_;
    }
    return dataLength;
}

int32_t
SocketApiUdp::Ipv4UdpRecvData(uint8_t *recvBuf, int32_t length, uint32_t timeoutMs)
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
        // 超时，此种情况下为udp无应答用例
        INFO_LOG << "UDP Timeout reached";
        return -2;
    }
    else {
        struct sockaddr_in clientAddr;
        socklen_t clientAddrLen = sizeof(clientAddr);
        recv_len = recvfrom(socket_ifd_, recvBuf, length, 0, (struct sockaddr *)&clientAddr, &clientAddrLen);
        if (recv_len < 0) {
            FAIL_LOG << "recvfrom fail";
            perror("recv()");
            return -1;
        }
        else if (recv_len == 0) {
            FAIL_LOG << "udp recv timeout";
            return -2;
        }
        else {
            // 处理接收到的数据
            // ...
            char ip[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &clientAddr.sin_addr, ip, sizeof(ip));
            DEBUG_LOG << "recv UDP ip addr " << ip << " port " << clientAddr.sin_port;
        }
    }

    return recv_len;
}

int32_t
SocketApiUdp::Ipv4UdpBindAddr(const char *ip, int32_t port)
{
    struct sockaddr_in serverAddr;
    memset(&serverAddr, 0, sizeof(serverAddr));

    while(1) {
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_port = htons(rand()%(65535-49152)+49152);//49152-65535
        serverAddr.sin_addr.s_addr = htonl(INADDR_ANY); // 使用任意可用的本地IP地址
        //udp_addr_.sin_addr.s_addr = inet_addr(ip);
        DEBUG_LOG << "bind ip " << ip << " port " << ntohs(serverAddr.sin_port);
        INFO_LOG << ntohs(serverAddr.sin_port);

        int bindResult = bind(socket_ifd_, (struct sockaddr *)&serverAddr, sizeof(serverAddr));
        if (bindResult < 0) {
            int err = errno;
            INFO_LOG << "Failed to bind socket to address, " << strerror(errno) << ", bind ip " << ip << " port " << ntohs(serverAddr.sin_port);
            if(err == EADDRINUSE) {
                continue;
            }
            return -1;
        }
        break;
    }
    memset(&udp_addr_, 0, sizeof(udp_addr_));
    udp_addr_.sin_family = AF_INET;
    udp_addr_.sin_port = htons(port);
    udp_addr_.sin_addr.s_addr = inet_addr(ip);
    DEBUG_LOG << "ip " << ip << " port " << port;

    return 0;
}

int32_t 
SocketApiUdp::Ipv4UdpClose()
{
    if (socket_ifd_ >= 0) {
        close(socket_ifd_);
        socket_ifd_ = -1;
    }
    tcp_link_ = false;
    return 0;
}

bool 
SocketApiUdp::GetLinkStatus()
{
    return tcp_link_;
}

