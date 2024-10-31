/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: main function definition
 */

#include <iostream>
#include <sys/socket.h>
#include <unistd.h>
#include "extwdg_udp_cm.h"

namespace hozon {
namespace netaos {
namespace extwdg {

int32_t
UdpConnector::Init()
{
    EW_INFO << "UDPCM::Init enter!";
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if(sockfd < 0) {
        EW_ERROR  << "Failed to create socket";
        return -1;
    }
    addr.sin_family = AF_INET;
    addr.sin_port = htons(host_port_);
    EW_INFO << "host_ip_ "<<host_ip_<<"host_port_"<<host_port_;
    if(inet_pton(AF_INET, host_ip_.c_str(), &(addr.sin_addr)) < 0) {
        EW_ERROR  << "Invalid host IP address.";
        return -1;
    }
    if(bind(sockfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        EW_ERROR  << "Failed to bind socket";
        return -1;
    }
    t1 = std::thread(std::bind(&UdpConnector::Recv, this));
    // t1.detach();
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(dest_port_);
    EW_INFO << "dest_ip_ "<<dest_ip_<<"dest_port_"<<dest_port_;
    if(inet_pton(AF_INET, dest_ip_.c_str(), &(dest_addr.sin_addr)) < 0) {
        EW_ERROR  << "Invalid dest IP address.";
        return -1;
    }

    init_flag = true;

    return 0;
}

void
UdpConnector::DeInit()
{
    stop_flag = true;
    MessageBuffer closemsg;
    closemsg.request = 0x4;
    sendto(sockfd, &closemsg, sizeof(closemsg), 0, (struct sockaddr*)&addr, sizeof(addr));
    t1.join();
    close(sockfd);
    sockfd = -1;
    init_flag = false;
    EW_INFO << "UdpConnector::DeInit success!";
}

int32_t
UdpConnector::Send(const MessageBuffer& sendmsg)
{
    EW_INFO << "UDPCM::Send enter!";
    ssize_t sendBytes = sendto(sockfd, &sendmsg, sizeof(sendmsg), 0, (struct sockaddr*)&dest_addr, sizeof(dest_addr));
    if(sendBytes < 0) {
        EW_ERROR  << "Faild to send data";
        return -1;
    }
    return 0;
}

void
UdpConnector::Recv()
{
    EW_INFO << "UDPCM::Recv enter!";
    MessageBuffer buffer;
    socklen_t addrlen = sizeof(addr);
    while(!stop_flag) {
        EW_INFO << "UDPCM::Recv loop enter!";
        memset(&buffer, 0, sizeof(buffer));

        ssize_t recvBytes = recvfrom(sockfd, &buffer, sizeof(buffer), 0, nullptr, nullptr);
        if(recvBytes < 0) {
            EW_ERROR  << "Faild to receive data";
            continue;
        }
        EW_INFO << "Recvhead is : " << buffer.head
                << "Recvseq is : "  << buffer.seq
                << "Recvreq is : "  << buffer.request
                << "Recvdata is : " << buffer.data;

        callback_(buffer);
    }
}

}  // namespace extwdg
}  // namespace netaos
}  // namespace hozon