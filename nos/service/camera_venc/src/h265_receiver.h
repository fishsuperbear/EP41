/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#ifndef H265_RECEIVER_H
#define H265_RECEIVER_H
#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "proto/soc/sensor_image.pb.h"
#include "cm/include/proxy.h"

namespace hozon {
namespace netaos {
namespace cameravenc {

class H265Receiver {
public:
    H265Receiver();
    ~H265Receiver();

    void SetTopics(std::vector<std::string>& topics);
    void Init();
    void Deinit();

    bool Get(hozon::soc::CompressedImage& h265_image);
    bool Get(std::string& topic, hozon::soc::CompressedImage& h265_image);

private:
    std::map<std::string, std::unique_ptr<hozon::netaos::cm::Proxy>> proxys_;
};

}
}
}
#endif