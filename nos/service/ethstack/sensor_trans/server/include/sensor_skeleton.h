#pragma once

#include <iostream>
#include <memory>
#include "cm/include/skeleton.h"

namespace hozon {
namespace netaos {
namespace sensor {
class Skeleton {
public:
    Skeleton(uint32_t domainID, std::string topic);
    ~Skeleton();
    // int Init();
    int Write(std::shared_ptr<void> data);
    int Deinit();
private:
    std::shared_ptr<hozon::netaos::cm::Skeleton> skeleton_;
};
}   // namespace sensor
}   // namespace netaos
}   // hozon