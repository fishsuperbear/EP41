#ifndef SOMEIP_CAPTURE_CONFIG_H
#define SOMEIP_CAPTURE_CONFIG_H
#pragma once

#include "network_capture/include/base_capture_config.h"
#include <vector>
#include <map>
namespace hozon {
namespace netaos {
namespace network_capture {

class SomeipFilterInfo : public BaseFilterInfo {
   public:
    
    std::map <std::uint32_t, std::string> topic_map;

    SomeipFilterInfo()
    : topic_map({}) { }

    static std::unique_ptr<std::vector<std::unique_ptr<SomeipFilterInfo>>> LoadConfig(std::string file_path);
};


}  // namespace network_capture
}  // namespace netaos
}  // namespace hozon

#endif