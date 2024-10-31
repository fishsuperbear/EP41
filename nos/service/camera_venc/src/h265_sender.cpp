/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#include "h265_sender.h"
#include "camera_venc_logger.h"
#include "idl/generated/cm_protobufPubSubTypes.h"

namespace hozon {
namespace netaos {
namespace cameravenc {

H265Sender::H265Sender() {}

H265Sender::~H265Sender() {}

void H265Sender::SetTopics(std::vector<std::string>& topics) {
    for (auto& topic : topics) {
        skeletons_[topic] = nullptr;
    }
}

void H265Sender::Init() {
    for (auto& it : skeletons_) {
        if (!it.second) {
            it.second = std::make_unique<hozon::netaos::cm::Skeleton>(std::make_shared<CmProtoBufPubSubType>());
            if (it.second->Init(0, it.first) < 0) {
                CAMV_ERROR << "Init cm proxy failed. Topic: " << it.first;
                it.second.reset();
            }
        }
    }
}

void H265Sender::Deinit() {
    for (auto& it : skeletons_) {
        if (!it.second) {
            it.second->Deinit();
            it.second.reset();
        }
    }
}

bool H265Sender::Put(const std::string& topic, hozon::soc::CompressedImage& h265_image) {

    if (!skeletons_[topic]) {
        CAMV_WARN << "Cm proxy is not inited. Topic: " << topic;
        return false;
    }

    if (!skeletons_[topic]->IsMatched()) {
        CAMV_INFO << "Cm proxy is not matched yet. Topic: " << topic;
        // return false;
    }

    std::shared_ptr<CmProtoBuf> cm_idl_data = std::make_shared<CmProtoBuf>();
    cm_idl_data->name(h265_image.GetTypeName());
    std::vector<char> data_vec;
    std::string data_str;
    if (!h265_image.SerializeToString(&data_str)) {
        CAMV_ERROR << "Serialize h265_image to string failed. Topic: " << topic;
        return false;
    }
    data_vec.resize(data_str.size());
    memcpy(data_vec.data(), data_str.data(), data_str.size());
    cm_idl_data->str(data_vec);

    CAMV_INFO << "Topic: " << topic << ", frame_type: " << h265_image.frame_type() << ", size: " << h265_image.data().size();
    if (skeletons_[topic]->Write(cm_idl_data) < 0) {
        CAMV_WARN << "Write data to cm failed. Topic: " << topic;
        return false;
    }

    return true;
}

}  // namespace cameravenc
}  // namespace netaos
}  // namespace hozon