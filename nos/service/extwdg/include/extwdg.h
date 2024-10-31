/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: extwdg
*/
#ifndef EXTWDG_H_
#define EXTWDG_H_

#include <memory>
#include <vector>
#include <cstdint>
#include "extwdg_transport.h"
#include "extwdg_logger.h"

namespace hozon {
namespace netaos {
namespace extwdg {

class Transport;

struct TransportInfo
{
    std::string dest_ip;
    int dest_port;
    std::string host_ip;
    int host_port;
    std::string protocol;
    TransportInfo():dest_ip("172.16.90.10")
    ,dest_port(23461)
    ,host_ip("172.16.90.11")
    ,host_port(23461)
    ,protocol("UDP") {}
};

class ExtWdg
{
public:
    ExtWdg(std::vector<TransportInfo> transinfo);
    ~ExtWdg() {}
    int32_t Init();
    void DeInit();
    void Run();
    void Stop();

private:
    std::vector<TransportInfo> trans_info;
    std::shared_ptr<Transport> transport_;
    bool  init_ = false;
};

}  // namespace extwdg
}  // namespace netaos
}  // namespace hozon

#endif // EXTWDG_H_