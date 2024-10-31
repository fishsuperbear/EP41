/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: extwdg
*/
#ifndef EXTWDG_UDP_CM_H_
#define EXTWDG_UDP_CM_H_

#include <cstring>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <functional>
#include <thread>
#include "extwdg_connector_base.h"
#include "extwdg_logger.h"
#include "extwdg_message.h"
#include "extwdg_transport.h"

namespace hozon {
namespace netaos {
namespace extwdg {

class UdpConnector : public Connector
{
public:
    UdpConnector(std::string dest_ip, int dest_port, std::string host_ip, int host_port, std::function<int32_t(const MessageBuffer&)> callback)
    : Connector(dest_ip, dest_port, host_ip, host_port)
    ,sockfd(-1)
    ,callback_(callback)
    ,host_ip_(host_ip)
    ,host_port_(host_port)
    ,dest_ip_(dest_ip)
    ,dest_port_(dest_port)
    {
        memset(&addr, 0, sizeof(addr));
    }
    ~UdpConnector() {}
    int32_t Init();
    void DeInit();
    int32_t Send(const MessageBuffer& sendmsg);
    void Recv();
    // uint32_t GetHead();
    // void testSend();

private:
    uint32_t head = 0xFACE;
    std::string host_ip_;
    std::string dest_ip_;
    int host_port_;
    int dest_port_;
    int sockfd;
    struct sockaddr_in addr;
    struct sockaddr_in dest_addr;
    bool init_flag = false;
    std::function<int32_t(const MessageBuffer&)> callback_;
    std::thread t1;
    bool stop_flag = false;
};

}  // namespace extwdg
}  // namespace netaos
}  // namespace hozon

#endif // EXTWDG_UDP_CM_H_