/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#ifndef H265_SENDER_H
#define H265_SENDER_H
#pragma once

#include <string>
#include <vector>
#include <map>
#include "proto/soc/sensor_image.pb.h"
#include "cm/include/skeleton.h"

namespace hozon {
namespace netaos {
namespace cameravenc {

class H265Sender {
public:
    H265Sender();
    ~H265Sender();

    void SetTopics(std::vector<std::string>& topics);
    void Init();
    void Deinit();

    bool Put(const std::string& topic, hozon::soc::CompressedImage& h265_image);

private:
    std::map<std::string, std::unique_ptr<hozon::netaos::cm::Skeleton>> skeletons_;
};

}
}
}
#endif