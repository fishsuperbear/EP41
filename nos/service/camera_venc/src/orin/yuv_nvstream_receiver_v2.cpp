/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#include "yuv_nvstream_receiver_v2.h"
// #include "NvSIPLCamera.hpp"
#include "camera_venc_logger.h"
#include "sensor/nvs_adapter/nvs_helper.h"
#include "sensor/nvs_consumer/CEncManager.h"
#include "sensor/nvs_consumer/CIpcConsumerChannel.hpp"

extern int8_t g_consumer_index;

namespace hozon {
namespace netaos {
namespace cameravenc {

// YuvBufWrapper::~YuvBufWrapper() {
//     // if (release) {
//     //     release(static_cast<hozon::netaos::nv::IEPPacket*>(packet), *static_cast<NvSciSyncFence*>(pre_fence), *static_cast<NvSciSyncFence*>(eof_fence));
//     // }
// }

YuvNvStreamReceiverV2::YuvNvStreamReceiverV2() {

    // // front camera: front 0
    // sensor_info_mapping_["/sensors/camera/camera_0"] = CamInfo{ 0, 3840, 2160 };
    // // front camera: front 1
    // sensor_info_mapping_["/sensors/camera/camera_1"] = CamInfo{ 1, 3840, 2160 };
    // // rear camera: rear
    // sensor_info_mapping_["/sensors/camera/camera_3"] = CamInfo{ 3, 1920, 1536 };

    // // side camera: front left
    // sensor_info_mapping_["/sensors/camera/camera_4"] = CamInfo{ 4, 1920, 1536 };
    // // side camera: front right
    // sensor_info_mapping_["/sensors/camera/camera_5"] = CamInfo{ 5, 1920, 1536 };
    // // side camera: rear left
    // sensor_info_mapping_["/sensors/camera/camera_6"] = CamInfo{ 6, 1920, 1536 };
    // // side camera: rear right
    // sensor_info_mapping_["/sensors/camera/camera_7"] = CamInfo{ 7, 1920, 1536 };

    // // fisheye camera: right
    // sensor_info_mapping_["/sensors/camera/camera_8"] = CamInfo{ 8, 1920, 1536 };
    // // fisheye camera: left
    // sensor_info_mapping_["/sensors/camera/camera_9"] = CamInfo{ 9, 1920, 1536 };
    // // fisheye camera: rear
    // sensor_info_mapping_["/sensors/camera/camera_10"] = CamInfo{ 10, 1920, 1536 };
    // // fisheye camera: front
    // sensor_info_mapping_["/sensors/camera/camera_11"] = CamInfo{ 11, 1920, 1536 };
}

YuvNvStreamReceiverV2::~YuvNvStreamReceiverV2() {}

void YuvNvStreamReceiverV2::SetTopics(std::vector<std::string>& topics) {
    for (auto it = sensor_info_mapping_.begin(); it != sensor_info_mapping_.end();) {
        bool set = false;

        for (auto& topic : topics) {
            if (topic == it->first) {
                set = true;
                break;
            }
        }

        if (!set) {
            it = sensor_info_mapping_.erase(it);
        } else {
            ++it;
        }
    }
}

void YuvNvStreamReceiverV2::SetSensorInfos(const SensorInfoMap& sensor_infos) {
    sensor_info_mapping_ = sensor_infos;
}

void YuvNvStreamReceiverV2::SetCallbacks(std::string topic, hozon::netaos::desay::EncConsumerCbs cbs) {
    cbs_map_[topic] = cbs;
}

int YuvNvStreamReceiverV2::Init() {
    std::map<uint32_t, hozon::netaos::desay::EncConsumerCbs> sensor_cbs_map;
    for (auto& it : cbs_map_) {
        sensor_cbs_map[sensor_info_mapping_[it.first].sensor_id] = it.second;
    }

    hozon::netaos::desay::CEncManager::Instance().SetSensorImageCbMap(sensor_cbs_map);
    // hozon::netaos::multicast::multicast_set_enc_cbs(sensor_cbs_map);
    // char const * argv[] = {
    //     "camera_venc",
    //     "-g",
    //     "XPC_F120_OX08B40_MAX96717_CPHY_x4",
    //     "-m",
    //     "\"0011 0 0 0\"",
    //     "-c",
    //     "\"enc\"",
    //     "-v",
    //     "4"};
    // int res = hozon::netaos::multicast::multicast_init(sizeof(argv) / sizeof(char const *), const_cast<char**>(argv));
    // if (res != 0) {
    //     CAMV_ERROR << "Init multicast with enc consumer failed. res: " << res;
    // }

    hozon::netaos::nv::NVSHelper::GetInstance().Init();
    // std::string ipc_channel = std::string("cam") + std::to_string(sensor_id) + "_recv" + std::to_string(channel_id);
    // NVS_LOG_INFO << "Init ipc channel " << ipc_channel;
    // _freq_checker = std::make_shared<SimpleFreqChecker>([](const std::string& name, double freq){
    //     NVS_LOG_INFO << "Check " << name << " frequency: " << freq << " Hz";
    // });

    CAMV_INFO << "consumer index : " << g_consumer_index;

    for (auto& it : sensor_info_mapping_) {
        nvsipl::SensorInfo sensor_info;
        sensor_info.id = it.second.sensor_id;
        sensor_info.vcInfo.resolution.width = it.second.resolution_width;
        sensor_info.vcInfo.resolution.height = it.second.resolution_height;
        CAMV_INFO << "Sensor id: " << sensor_info.id << ", resolution: " << sensor_info.vcInfo.resolution.width << " x " << sensor_info.vcInfo.resolution.height << ", topic: " << it.first;
        auto consumer =
            std::make_shared<hozon::netaos::desay::CIpcConsumerChannel>(hozon::netaos::nv::NVSHelper::GetInstance().sci_buf_module, hozon::netaos::nv::NVSHelper::GetInstance().sci_sync_module,
                                                                        &sensor_info, hozon::netaos::desay::ENC_CONSUMER, it.second.sensor_id, g_consumer_index);

        hozon::netaos::desay::ConsumerConfig consumer_config{false, false};
        consumer->SetConsumerConfig(consumer_config);

        auto status = consumer->CreateBlocks(nullptr);
        CHK_STATUS_AND_RETURN(status, "Master CreateBlocks");

        status = consumer->Connect();
        CHK_STATUS_AND_RETURN(status, "CMaster: Channel connect.");

        status = consumer->InitBlocks();
        CHK_STATUS_AND_RETURN(status, "InitBlocks");

        status = consumer->Reconcile();
        CHK_STATUS_AND_RETURN(status, "Channel Reconcile");

        // static_cast<CEncConsumer*>(consumer.m_upConsumer.get())->SetOnPacketCallback(
        //         std::bind(&PacketRecvCallback, static_cast<CEncConsumer*>(consumer.m_upConsumer.get()), std::placeholders::_1));
        consumer->Start();

        proxys_[it.first] = consumer;
    }

    return 0;
}

void YuvNvStreamReceiverV2::Deinit() {
    // hozon::netaos::multicast::multicast_quit();
    for (auto& i : proxys_) {
        if (nullptr != i.second) {
            i.second->Stop();
        }
    }
}

std::shared_ptr<struct YuvBufWrapper> YuvNvStreamReceiverV2::Get() {
    return nullptr;
}

std::shared_ptr<YuvBufWrapper> YuvNvStreamReceiverV2::Get(std::string& topic) {

    // if (!proxys_[topic]) {
    //     CAMV_WARN << "ns stream consumer is not inited. Topic: " << topic;
    //     return nullptr;
    // }

    // if (!proxys_[topic]->_nvs_img_consumer.Ready()) {
    //     // CAMV_INFO << "Cm proxy is not matched yet. Topic: " << topic;
    //     return false;
    // }

    std::shared_ptr<YuvBufWrapper> buf_wrapper;
    // hozon::netaos::nv::IEPPacket* packet = proxys_[topic]->GetPacket();
    // if (packet) {
    //     buf_wrapper->packet = std::make_share<YuvBufWrapper>()
    //     buf_wrapper->buf_obj = nullptr;
    //     buf_wrapper->pre_fence = nullptr;
    //     buf_wrapper->eof_fence = nullptr;
    //     buf_wrapper->release = std::bind(&NVSBlockIEPConsumer::PacketConsumed, &proxys_[topic]->_nvs_img_consumer, std::placeholders::_1, std::placeholder::_2, std::placeholder::_3);
    // }

    return buf_wrapper;
}

}  // namespace cameravenc
}  // namespace netaos
}  // namespace hozon