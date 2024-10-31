/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: extwdg
*/
#ifndef EXTWDG_CONNECTOR_BASE_H_
#define EXTWDG_CONNECTOR_BASE_H_

#include <cstdint>
#include <string>
#include "extwdg_message.h"

namespace hozon {
namespace netaos {
namespace extwdg {

class Connector
{
public:
    Connector(std::string dest_ip, int dest_port, std::string host_ip, int host_port)
    :host_ip_(host_ip)
    ,dest_ip_(dest_ip)
    ,host_port_(host_port)
    ,dest_port_(dest_port)
    {}
    ~Connector() {}
    virtual int32_t Init() = 0;
    virtual void DeInit() = 0;
    virtual int32_t Send(const MessageBuffer& sendmsg) = 0;
    virtual void Recv() = 0;
    uint32_t GetHead() {return head;}
    uint32_t CheckInit() {return init_flag;}

private:
    uint32_t head;
    std::string host_ip_;
    std::string dest_ip_;
    int host_port_;
    int dest_port_;
    bool init_flag;
};

}  // namespace extwdg
}  // namespace netaos
}  // namespace hozon

#endif // EXTWDG_CONNECTOR_BASE_H_