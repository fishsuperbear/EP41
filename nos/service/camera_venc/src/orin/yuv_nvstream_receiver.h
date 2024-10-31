/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#ifndef YUV_NVSTREAM_RECEIVER_H
#define YUV_NVSTREAM_RECEIVER_H
#pragma once

#include <string>
#include <vector>
#include <map>
#include <functional>
#include "nvs_iep_adapter.h"

namespace hozon {
namespace netaos {
namespace cameravenc {

struct YuvBufWrapper {
    void *pre_fence = nullptr;
    void *eof_fence = nullptr;
    void *packet = nullptr;

    std::function<void(void *packet, void *pref_fence, void *eof_fence)> release;

    ~YuvBufWrapper();
};

class YuvNvStreamReceiver {
public:
    YuvNvStreamReceiver();
    ~YuvNvStreamReceiver();

    void SetTopics(std::vector<std::string>& topics);
    int Init();
    void Deinit();

    void SetCallbacks(std::string topic, hozon::netaos::nv::IEPConsumerCbs cbs);
    std::shared_ptr<struct YuvBufWrapper> Get();
    std::shared_ptr<struct YuvBufWrapper> Get(std::string& topic);

private:
    std::map<std::string, std::shared_ptr<hozon::netaos::nv::NVSIEPAdapter>> proxys_;
    std::map<std::string, std::string> topic_nvstream_channel_mapping_;
    std::map<std::string, hozon::netaos::nv::IEPConsumerCbs> cbs_map_;
};

}
}
}
#endif