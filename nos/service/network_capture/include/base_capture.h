/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#ifndef BASE_CAPTURE_H
#define BASE_CAPTURE_H
#pragma once

#include "network_capture/include/base_capture_config.h"
#include <netinet/ip.h>
#include <thread>
#include <pcap.h>

namespace hozon {
namespace netaos {
namespace network_capture {
class BaseCapture {
   public:
    virtual std::int32_t Init() = 0;
    virtual std::int32_t DeInit() = 0;
    virtual std::int32_t Run() = 0;
    virtual std::int32_t Stop() = 0;
   protected:
    virtual void capPacket(std::string eth_name, std::string filter_exp) = 0;
    virtual void process_packet(const u_char *packet, pcap_pkthdr& header) = 0;
};

}  // namespace network_capture
}  // namespace netaos
}  // namespace hozon

#endif