/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#include "yuv_receiver.h"
#include "camera_venc_logger.h"
#include "idl/generated/cm_protobufPubSubTypes.h"

namespace hozon {
namespace netaos {
namespace cameravenc {

YuvReceiver::YuvReceiver() {

}

YuvReceiver::~YuvReceiver() {

}

void YuvReceiver::SetTopics(std::vector<std::string>& topics) {
    for (auto& topic : topics) {
        proxys_[topic] = nullptr;
    }
}

void YuvReceiver::Init() {
    for (auto& it : proxys_) {
        if (!it.second) {
            it.second = std::make_unique<hozon::netaos::cm::Proxy>(std::make_shared<CmProtoBufPubSubType>());
            CAMV_INFO << "Init proxy. Topic: " << it.first;
            if (it.second->Init(0, it.first) < 0) {
                CAMV_ERROR << "Init cm proxy failed. Topic: " << it.first;
                it.second.reset();
            }
        }
    }
}

void YuvReceiver::Deinit() {
    for (auto& it : proxys_) {
        if (!it.second) {
            it.second->Deinit();
            it.second.reset();
        }
    }
}

bool YuvReceiver::Get(hozon::soc::Image& yuv_image) {
    return false;
}

bool YuvReceiver::Get(std::string& topic, hozon::soc::Image& yuv_image) {

    if (!proxys_[topic]) {
        CAMV_WARN << "Cm proxy is not inited. Topic: " << topic;
        return false;
    }

    if (!proxys_[topic]->IsMatched()) {
        // CAMV_INFO << "Cm proxy is not matched yet. Topic: " << topic;
        return false;
    }

    std::shared_ptr<CmProtoBuf> cm_idl_data = std::make_shared<CmProtoBuf>();
    if (proxys_[topic]->Take(cm_idl_data, 1000) < 0) {
        CAMV_WARN << "Take data from cm failed. Topic: " << topic;
        return false;
    }

    if (!yuv_image.ParseFromArray(cm_idl_data->str().data(), cm_idl_data->str().size())) {
        CAMV_WARN << "Parse protobuf failed. Topic: " << topic;
        return false;
    }

    return true;
}

}
}
}