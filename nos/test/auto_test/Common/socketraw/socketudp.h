/*
 * Copyright (c) Hozon SOC Co., Ltd. 2023-2023. All rights reserved.
 *
 * Description: udp socket
 */
#ifndef SOCKET_UDP_H
#define SOCKET_UDP_H

#include <string>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>


class SocketApiUdp {
public:
    SocketApiUdp();
    int32_t CreateSocket(int32_t domain, int32_t type, int32_t protocol);
    int32_t Ipv4UdpClose();
    bool GetLinkStatus();

    int32_t Ipv4UdpSendData(uint8_t *data, int32_t dataLength);
    int32_t Ipv4UdpRecvData(uint8_t *recvBuf, int32_t length, uint32_t timeoutMs);
    int32_t Ipv4UdpBindAddr(const char *ip, int32_t port);

private:
    int32_t socket_ifd_;
    bool tcp_link_;


    sockaddr_in udp_addr_;
    //int32_t protocol_type_;
};


#endif

