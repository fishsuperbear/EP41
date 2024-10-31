/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#ifndef BASE_CAPTURE_CONFIG_H
#define BASE_CAPTURE_CONFIG_H
#pragma once

#include <string>
#include <memory>
#include <vector>
#include <fstream>
#include <queue>
#include <mutex>
#include "network_capture/include/network_logger.h"

namespace hozon {
namespace netaos {
namespace network_capture {

class BaseFilterInfo {
   public:
    std::string eth_name;
    std::string protocol;
    std::string src_host;
    std::string dst_host;
    std::string src_port;
    std::string dst_port;
    std::vector<uint16_t> ports;

    BaseFilterInfo() 
    : eth_name("")
    , protocol("")
    , src_host("")
    , dst_host("")
    , src_port("")
    , dst_port("") 
    , ports({}) { }

    ~BaseFilterInfo() = default;
};


}  // namespace network_capture
}  // namespace netaos
}  // namespace hozon

#endif