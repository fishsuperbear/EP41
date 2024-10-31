/*
 * Copyright (c) Hozon SOC Co., Ltd. 2023-2023. All rights reserved.
 *
 * Description: tcp socket
 */
#ifndef SOCKET_H
#define SOCKET_H

#include <string>
#include <vector>
#include <sys/socket.h>


class SocketApi {
public:
    SocketApi();
    int32_t CreateSocket(int32_t domain, int32_t type, int32_t protocol);
    int32_t Ipv4TcpConnect(const char *ip, uint16_t port, const char *ifName);
    int32_t Ipv4TcpSendData(uint8_t *data, int32_t dataLength);
    int32_t Ipv4TcpRecvData(uint8_t *recv, int32_t length, uint32_t timeoutMs);
    int32_t Ipv4TcpClose();
    int32_t Ipv4TcpShutDown();

    int32_t SocketCanConfig(const char *ifName, bool noblock);
    int32_t SocketCanSend(uint32_t canid, uint8_t *data, int32_t size);
    int32_t SocketCanRead(uint32_t *canid, uint8_t *data, int32_t size);
    int32_t SocketCanRecv(uint32_t *canid, uint8_t *data, int32_t size, uint32_t timeoutms);
    bool GetLinkStatus();

private:
    int32_t Connect(int32_t sockfd, struct sockaddr *addr, socklen_t addrlen);
    int32_t socket_ifd_;
    bool tcp_link_;
};


#endif

