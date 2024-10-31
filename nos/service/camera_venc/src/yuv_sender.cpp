/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#include "yuv_sender.h"
#include "camera_venc_logger.h"

namespace hozon {
namespace netaos {
namespace cameravenc {

YuvSender::YuvSender() {

}

YuvSender::~YuvSender() {

}

void YuvSender::SetTopics(std::vector<std::string>& topics) {

}

void YuvSender::Init() {

}

void YuvSender::Deinit() {

}

bool YuvSender::Put(const std::string& topic, hozon::soc::Image& yuv_image) {
    return false;
}

}
}
}