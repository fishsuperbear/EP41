/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: main function definition
 */

#include "extwdg_transport.h"
#include "extwdg_logger.h"
#include <cstdint>

namespace hozon {
namespace netaos {
namespace extwdg {

Transport::Transport(const std::vector<TransportInfo>& trans_info)
{
    // EW_INFO << "Transport::Transport enter!";
    std::function<int32_t(const MessageBuffer&)> sendptr = std::bind(&Transport::Send, this, std::placeholders::_1);
    message_ = std::make_shared<Message>(sendptr);
    CreateConnectors(trans_info);
}

Transport::~Transport()
{
}

int32_t
Transport::Init()
{
    message_->RegisterCallback(getmessagecallback_);
    int32_t res = message_->Init();
    if(res == -1) {
        EW_ERROR << "Message::Init failed!";
        return -1;
    }
    EW_INFO << "Transport::Init exit!";
    return 0;
}

void
Transport::DeInit()
{
    EW_INFO << "Transport::DeInit deinit!";
    message_->DeInit();
    return;
}

int32_t
Transport::CreateConnectors(const std::vector<TransportInfo>& trans_info)
{
    int32_t ret = -1;
    for(auto it = trans_info.begin(); it != trans_info.end(); ++it) {
        if("UDP" == it->protocol) {
            connectors.emplace_back(std::make_shared<UdpConnector>(it->dest_ip, it->dest_port, it->host_ip, it->host_port, std::bind(&Transport::Recv, this, std::placeholders::_1)));
        }
    }
    ret = 0;
    return ret;
}

int32_t
Transport::Connect()
{
    int32_t res = -1;
    EW_INFO << "Transport::Connect enter!";
    for(uint32_t i = 0; i < connectors.size(); ++i) {
        connectors[i]->Init();
    }
    res = 0;
    return res;
}

int32_t
Transport::DisConnect()
{
    int32_t res = -1;
    EW_INFO << "Transport::DisConnect enter!";
    for(uint32_t i = 0; i < connectors.size(); ++i) {
        connectors[i]->DeInit();
    }
    res = 0;
    return res;
}

int32_t
Transport::Send(const MessageBuffer& sendmsg)
{
    // EW_INFO << "Transport::Send enter!";
    int32_t res = -1;
    for(uint32_t i = 0; i < connectors.size(); ++i) {
        if(0xFACE == sendmsg.head) {
                connectors[i]->Send(sendmsg);

        }
    }
    return 0;
}

int32_t 
Transport::Recv(const MessageBuffer& recvmsg)
{
    // EW_INFO << "Transport::Recv enter!";
    int32_t res = -1;
    for(uint32_t i = 0; i < connectors.size(); ++i) {
        if(0xFACE == recvmsg.head ) {
            getmessagecallback_(recvmsg);
            res = 0;
        }
    }
    return res;
}

}  // namespace extwdg
}  // namespace netaos
}  // namespace hozon