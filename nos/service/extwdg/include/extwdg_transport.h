/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: extwdg
*/
#ifndef EXTWDG_TRANSPORT_H_
#define EXTWDG_TRANSPORT_H_

#include <memory>
#include <vector>
#include <cstdint>
#include "extwdg_udp_cm.h"
#include "extwdg_message.h"
#include "extwdg.h"
#include "extwdg_logger.h"

namespace hozon {
namespace netaos {
namespace extwdg {

struct TransportInfo;
// typedef  std::function<int32_t(const MessageBuffer&)> RecvCallback;

class Transport
{
public:
    Transport(const std::vector<TransportInfo>& trans_info);
    ~Transport();
    int32_t Init();
    void DeInit();
    int32_t Connect();
    int32_t DisConnect();
    int32_t Send(const MessageBuffer& sendmsg);
    int32_t Recv(const MessageBuffer& recvmsg);

private:
    int32_t CreateConnectors(const std::vector<TransportInfo>& trans_info);

private:
    // std::vector<Connector*> connectors;
    std::vector<std::shared_ptr<Connector>> connectors;
    std::shared_ptr<Message> message_;
    std::function<void(const MessageBuffer&)> getmessagecallback_ = nullptr;
};

}  // namespace extwdg
}  // namespace netaos
}  // namespace hozon

#endif // EXTWDG_TRANSPORT_H_