/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#include "camera_venc.h"
#include <fstream>
#include "codec/include/codec_def.h"
#include "em/include/exec_client.h"
#include "em/include/proctypes.h"

#ifdef BUILD_FOR_ORIN
#include "sensor/nvs_adapter/nvs_utility.h"
#include "sensor/nvs_consumer/CEncManager.h"
#endif

#include "camera_venc_config.h"
#include "camera_venc_logger.h"

extern bool g_write_to_file;
extern std::string g_conf_path;

namespace hozon {
namespace netaos {
namespace cameravenc {

// const bool PollMode = false;

CameraVenc::CameraVenc() : work_mode_(kWorkNone), write_file_(false), stopped_(false) {}

CameraVenc::~CameraVenc() {}

void CameraVenc::Init() {

    auto cfg = CameraVencConfig::LoadConfig(g_conf_path);
    if (!cfg) {
        CAMV_CRITICAL << "Load camera venc config failed.";
        return;
    }

    // Print config informations.
    std::ostringstream oss;
    oss << "Config codec_type: " << cfg->codec_type << ", write_file: " << cfg->write_file << ", uhp_mode: " << cfg->uhp_mode;
    oss << ", selected sensor ids:";
    for (auto id : cfg->sensor_ids) {
        oss << " " << id;
    }
    CAMV_INFO << oss.str();

    for (auto it : cfg->sensor_infos) {
        CAMV_INFO << "Sensor_id: " << it.sensor_id << ", resolution: " << it.resolution_width << "x" << it.resolution_height << ", yuv_topic: " << it.yuv_topic << ", enc_topic: " << it.enc_topic;
    }

    // Report em running.
    // std::shared_ptr<hozon::netaos::em::ExecClient> execli = std::make_shared<hozon::netaos::em::ExecClient>();
    // execli->ReportState(hozon::netaos::em::ExecutionState::kRunning);

    work_mode_ = kWorkEncoder;
    // write_file_ = cfg->write_file;
    write_file_ = g_write_to_file;
    std::vector<std::string> enc_topics;

    for (auto it : cfg->sensor_infos) {
        sensor_info_mapping_[it.yuv_topic] = it;

        // init file handle
        if (write_file_) {
            size_t delm = it.yuv_topic.rfind("/");
            std::string file_name = it.yuv_topic.substr(delm + 1, it.yuv_topic.size() - delm - 1);
            file_name += ".265";
            file_map_[it.yuv_topic] = std::make_unique<std::ofstream>(file_name, std::ios::binary | std::ios::out);
        }
    }

    // Select sensors that need process.
    for (auto it = sensor_info_mapping_.begin(); it != sensor_info_mapping_.end();) {
        bool set = false;
        for (auto id : cfg->sensor_ids) {
            if (it->second.sensor_id == id) {
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

    // Get memory layout of all camera
    std::map<uint32_t, uint32_t> src_layout_map;
    for (auto it : cfg->sensor_infos) {
        src_layout_map[it.sensor_id] = it.src_layout;
    }

    // Define topic mapping between yuv and h265.
    // front camera: front 0
    // sensor_info_mapping_["/sensors/camera/camera_0"].enc_topic = "/sensors/camera/encoded_camera_0";
    // sensor_info_mapping_["/sensors/camera/camera_0"].encoder_cfg_file = "/app/conf/camera_venc/venc_h265_front.cfg";
    // // front camera: front 1
    // sensor_info_mapping_["/sensors/camera/camera_1"].enc_topic = "/sensors/camera/encoded_camera_1";
    // sensor_info_mapping_["/sensors/camera/camera_1"].encoder_cfg_file = "/app/conf/camera_venc/venc_h265_front.cfg";
    // // rear camera: rear
    // sensor_info_mapping_["/sensors/camera/camera_3"].enc_topic = "/sensors/camera/encoded_camera_3";
    // sensor_info_mapping_["/sensors/camera/camera_3"].encoder_cfg_file = "/app/conf/camera_venc/venc_h265_rear.cfg";

    // // side camera: front left
    // sensor_info_mapping_["/sensors/camera/camera_4"].enc_topic = "/sensors/camera/encoded_camera_4";
    // sensor_info_mapping_["/sensors/camera/camera_4"].encoder_cfg_file = "/app/conf/camera_venc/venc_h265_side.cfg";
    // // side camera: front right
    // sensor_info_mapping_["/sensors/camera/camera_5"].enc_topic = "/sensors/camera/encoded_camera_5";
    // sensor_info_mapping_["/sensors/camera/camera_5"].encoder_cfg_file = "/app/conf/camera_venc/venc_h265_side.cfg";
    // // side camera: rear left
    // sensor_info_mapping_["/sensors/camera/camera_6"].enc_topic = "/sensors/camera/encoded_camera_6";
    // sensor_info_mapping_["/sensors/camera/camera_6"].encoder_cfg_file = "/app/conf/camera_venc/venc_h265_side.cfg";
    // // side camera: rear right
    // sensor_info_mapping_["/sensors/camera/camera_7"].enc_topic = "/sensors/camera/encoded_camera_7";
    // sensor_info_mapping_["/sensors/camera/camera_7"].encoder_cfg_file = "/app/conf/camera_venc/venc_h265_side.cfg";

    // // fisheye camera: right
    // sensor_info_mapping_["/sensors/camera/camera_8"].enc_topic = "/sensors/camera/encoded_camera_8";
    // sensor_info_mapping_["/sensors/camera/camera_8"].encoder_cfg_file = "/app/conf/camera_venc/venc_h265_fisheye.cfg";
    // // fisheye camera: left
    // sensor_info_mapping_["/sensors/camera/camera_9"].enc_topic = "/sensors/camera/encoded_camera_9";
    // sensor_info_mapping_["/sensors/camera/camera_9"].encoder_cfg_file = "/app/conf/camera_venc/venc_h265_fisheye.cfg";
    // // fisheye camera: rear
    // sensor_info_mapping_["/sensors/camera/camera_10"].enc_topic = "/sensors/camera/encoded_camera_10";
    // sensor_info_mapping_["/sensors/camera/camera_10"].encoder_cfg_file = "/app/conf/camera_venc/venc_h265_fisheye.cfg";
    // // fisheye camera: front
    // sensor_info_mapping_["/sensors/camera/camera_11"].enc_topic = "/sensors/camera/encoded_camera_11";
    // sensor_info_mapping_["/sensors/camera/camera_11"].encoder_cfg_file = "/app/conf/camera_venc/venc_h265_fisheye.cfg";

    for (auto it = sensor_info_mapping_.begin(); it != sensor_info_mapping_.end(); ++it) {
        h265_yuv_mapping_[it->second.enc_topic] = it->first;
        enc_topics.push_back(it->second.enc_topic);
    }

    // Set topic for receiver and sender.
    h265_sender_.SetTopics(enc_topics);
    // yuv_receiver_.SetTopics(yuv_topics);

#ifdef BUILD_FOR_ORIN
    yuv_receiver_.SetSensorInfos(sensor_info_mapping_);
    // hozon::netaos::desay::CLogger::GetInstance().SetLogLevel(hozon::netaos::desay::CLogger::LEVEL_DEBUG);
    hozon::netaos::desay::CEncManager::Instance().SetCodecType(cfg->codec_type);
    hozon::netaos::desay::CEncManager::Instance().SetUhpMode(cfg->uhp_mode);
    hozon::netaos::desay::CEncManager::Instance().SetSrcLayout(src_layout_map);
    hozon::netaos::desay::CEncManager::Instance().SetFrameSampling(cfg->frame_sampling);

    for (auto& it : sensor_info_mapping_) {
        // hozon::netaos::nv::EncConsumerCbs cbs;
        // cbs.packet_cb = std::bind(&CameraVenc::OnYuvAvailable, this, it.first, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
        // cbs.get_buf_attr_cb = std::bind(&CameraVenc::OnGetBufAttrs, this, it.first, std::placeholders::_1);
        // cbs.get_waiter_attr_cb = std::bind(&CameraVenc::OnGetWaiterAttrs, this, it.first, std::placeholders::_1);
        // cbs.get_signaler_attr_cb = std::bind(&CameraVenc::OnGetSignalerAttrs, this, it.first, std::placeholders::_1);
        // cbs.set_signal_obj_cb = std::bind(&CameraVenc::OnSetSignalerObj, this, it.first, std::placeholders::_1);
        // cbs.set_waiter_obj_cb = std::bind(&CameraVenc::OnSetWaiterObj, this, it.first, std::placeholders::_1);
        // cbs.set_buf_attr_cb = std::bind(&CameraVenc::OnSetBufAttrs, this, it.first, std::placeholders::_1, std::placeholders::_2);
        // cbs.set_buf_obj_cb = std::bind(&CameraVenc::OnSetBufObj, this, it.first, std::placeholders::_1);
        // yuv_receiver_.SetCallbacks(it.first, cbs);
        hozon::netaos::desay::EncConsumerCbs cbs;
        // cbs.packet_cb = std::bind(&CameraVenc::OnYuvAvailable, this, it.first, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
        // cbs.get_buf_attr_cb = std::bind(&CameraVenc::OnGetBufAttrs, this, it.first, std::placeholders::_1);
        // cbs.get_waiter_attr_cb = std::bind(&CameraVenc::OnGetWaiterAttrs, this, it.first, std::placeholders::_1);
        // cbs.get_signaler_attr_cb = std::bind(&CameraVenc::OnGetSignalerAttrs, this, it.first, std::placeholders::_1);
        // cbs.set_signal_obj_cb = std::bind(&CameraVenc::OnSetSignalerObj, this, it.first, std::placeholders::_1);
        // cbs.set_waiter_obj_cb = std::bind(&CameraVenc::OnSetWaiterObj, this, it.first, std::placeholders::_1);
        // cbs.set_buf_attr_cb = std::bind(&CameraVenc::OnSetBufAttrs, this, it.first, std::placeholders::_1, std::placeholders::_2);
        // cbs.set_buf_obj_cb = std::bind(&CameraVenc::OnSetBufObj, this, it.first, std::placeholders::_1);
        cbs.encoded_image_cb = std::bind(&CameraVenc::OnH265Aailable, this, it.first, std::placeholders::_1);
        yuv_receiver_.SetCallbacks(it.first, cbs);
    }
#endif
    // // Init encoders.
    // for (auto& it : sensor_info_mapping_) {
    //     CAMV_INFO << "Init encoder with config file: " << it.second.encoder_cfg_file;
    //     encoder_map_[it.first] = hozon::netaos::codec::EncoderFactory::create(std::unordered_map<std::string, std::string>());
    //     encoder_map_[it.first]->Init(it.second.encoder_cfg_file);
    // }

    // Initiliaze receiver and sender using above topics.
    yuv_receiver_.Init();
    h265_sender_.Init();

    // Init encoders. TODO.
    CAMV_INFO << "Camera venc initialize end.";
}

void CameraVenc::Deinit() {
    CAMV_INFO << "Camera venc deinitialize start.";

    yuv_receiver_.Deinit();

    // Report em terminating.
    // std::shared_ptr<hozon::netaos::em::ExecClient> execli = std::make_shared<hozon::netaos::em::ExecClient>();
    // execli->ReportState(hozon::netaos::em::ExecutionState::kTerminating);
    CAMV_INFO << "Camera venc deinitialize end.";

    for (auto& v : file_map_) {
        v.second->close();
    }
}

void CameraVenc::Start() {

    if (work_mode_ == kWorkEncoder) {
#ifndef BUILD_FOR_ORIN
        // Start threads to encode pictures. Each topic has its own thread because encoding process is a time-costed opration.
        // for (auto& it : sensor_info_mapping_) {
        //     std::thread th(std::bind(&CameraVenc::ProcessTopic, this, it.first, it.second.enc_topic));
        //     work_threads_.push_back(std::move(th));
        // }
#endif
    } else if (work_mode_ == kWorkDecoder) {
        // Start threads to encode pictures. Each topic has its own thread because encoding process is a time-costed opration.
        // TODO
    } else {
        CAMV_WARN << "Work mode is none.";
    }
}

void CameraVenc::Stop() {
    stopped_ = true;

    // Deinit encoders.
    for (auto& it : sensor_info_mapping_) {
        encoder_map_[it.first].reset();
    }

    // Deinit decoders. TODO.

    for (auto& th : work_threads_) {
        th.join();
    }
}

#ifndef BUILD_FOR_ORIN
void CameraVenc::ProcessTopic(std::string yuv_topic, std::string enc_topic) {
    while (!stopped_) {
        hozon::soc::Image yuv_image;
        hozon::soc::CompressedImage h265_image;
        // Get one yuv from yuv receiver.
        if (yuv_receiver_.Get(yuv_topic, yuv_image)) {
            // Encode yuv to h265.
            hozon::netaos::codec::FrameType frame_type;
            std::string h265;
            auto before_encode = std::chrono::system_clock::now();
            auto res = encoder_map_[yuv_topic]->Process(yuv_image.data(), h265, frame_type);
            auto after_encode = std::chrono::system_clock::now();
            CAMV_DEBUG << "topic: " << yuv_topic << ", encode result: " << res << ", in size: " << yuv_image.data().size() << ", out size: " << h265.size() << ", type: " << frame_type
                       << ", time costed(ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(after_encode - before_encode).count();
            // std::ofstream ofs("fe_8.265", std::ios::binary | std::ios::app | std::ios::out);
            // ofs.write(h265.data(), h265.size());

            // Put h265 to sender.
            h265_image.set_data(h265);
            h265_sender_.Put(enc_topic, h265_image);
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(1000));
        }
    }
}
#else
int32_t CameraVenc::OnYuvAvailable(std::string topic, hozon::netaos::nv::IEPPacket* packet, NvSciSyncFence& pre_fence, NvSciSyncFence& eof_fence) {
    if (encoder_map_.find(topic) != encoder_map_.end()) {
        // Construct nv specific buf struct.
        hozon::netaos::codec::EncoderBufNvSpecific buf;
        buf.buf_obj = static_cast<void*>(packet->nv_sci_buf);
        buf.pre_fence = static_cast<void*>(&pre_fence);
        buf.eof_fence = static_cast<void*>(&eof_fence);

        std::string h265;
        hozon::netaos::codec::FrameType frame_type;
        auto before_encode = std::chrono::system_clock::now();
        auto res = encoder_map_[topic]->SetEncoderParam(hozon::netaos::codec::kEncoderParam_Nv_Prefence, static_cast<void*>(buf.pre_fence));
        if (res != hozon::netaos::codec::kEncodeSuccess) {
            CAMV_ERROR << "cannot set prefence to iep for topic: " << topic;
            return -1;
        }
        res = encoder_map_[topic]->Process(static_cast<void*>(&buf), h265, frame_type);
        auto after_encode = std::chrono::system_clock::now();
        CAMV_DEBUG << "topic: " << topic << ", encode result: " << res << ", out size: " << h265.size() << ", type: " << frame_type
                   << ", time costed(ms): " << std::chrono::duration_cast<std::chrono::milliseconds>(after_encode - before_encode).count();

        // std::ofstream ofs("camera_4.265", std::ios::binary | std::ios::app | std::ios::out);
        // ofs.write(h265.data(), h265.size());

        // Put h265 to sender.
        hozon::soc::CompressedImage h265_image;
        h265_image.set_data(h265);
        h265_sender_.Put(sensor_info_mapping_[topic].enc_topic, h265_image);
    }
    return hozon::netaos::nv::kPacketConsumed;
}

int32_t CameraVenc::OnGetBufAttrs(std::string topic, NvSciBufAttrList& buf_attrs) {
    CAMV_INFO << "Get buf attrs on topic: " << topic;
    if (encoder_map_.find(topic) != encoder_map_.end()) {
        void* vp = static_cast<void*>(buf_attrs);
        return encoder_map_[topic]->GetEncoderParam(hozon::netaos::codec::kEncoderParam_Nv_BufAttrs, &vp);
    }

    return -1;
}

int32_t CameraVenc::OnGetWaiterAttrs(std::string topic, NvSciSyncAttrList& waiter_attrs) {
    CAMV_INFO << "Get waiter attrs on topic: " << topic;
    if (encoder_map_.find(topic) != encoder_map_.end()) {
        void* vp = static_cast<void*>(waiter_attrs);
        return encoder_map_[topic]->GetEncoderParam(hozon::netaos::codec::kEncoderParam_Nv_WaiterAttrs, &vp);
    }

    return -1;
}

int32_t CameraVenc::OnGetSignalerAttrs(std::string topic, NvSciSyncAttrList& signaler_attrs) {
    CAMV_INFO << "Get signaler attrs on topic: " << topic;
    if (encoder_map_.find(topic) != encoder_map_.end()) {
        void* vp = static_cast<void*>(signaler_attrs);
        return encoder_map_[topic]->GetEncoderParam(hozon::netaos::codec::kEncoderParam_Nv_SignalerAttrs, &vp);
    }

    return -1;
}

int32_t CameraVenc::OnSetSignalerObj(std::string topic, NvSciSyncObj signaler_obj) {
    CAMV_INFO << "Set signaler object on topic: " << topic;
    if (encoder_map_.find(topic) != encoder_map_.end()) {
        return encoder_map_[topic]->SetEncoderParam(hozon::netaos::codec::kEncoderParam_Nv_SignalerObj, static_cast<void*>(signaler_obj));
    }

    return -1;
}

int32_t CameraVenc::OnSetWaiterObj(std::string topic, NvSciSyncObj waiter_obj) {
    CAMV_INFO << "Set waiter object on topic: " << topic;
    if (encoder_map_.find(topic) != encoder_map_.end()) {
        return encoder_map_[topic]->SetEncoderParam(hozon::netaos::codec::kEncoderParam_Nv_WaiterObj, static_cast<void*>(waiter_obj));
    }

    return -1;
}

int32_t CameraVenc::OnSetBufAttrs(std::string topic, int32_t elem_type, NvSciBufAttrList buf_attrs) {
    CAMV_INFO << "Set buf attrs of elem_type: " << hozon::netaos::log::LogHex32{static_cast<uint32_t>(elem_type)} << " on topic: " << topic;
    if ((elem_type == ELEMENT_NAME_IMAGE) && (encoder_map_.find(topic) != encoder_map_.end())) {
        NvSciBufAttrKeyValuePair keyVals[] = {{NvSciBufImageAttrKey_PlaneCount, NULL, 0},
                                              {NvSciBufImageAttrKey_PlaneColorFormat, NULL, 0},
                                              {NvSciBufImageAttrKey_PlaneWidth, NULL, 0},
                                              {NvSciBufImageAttrKey_PlaneHeight, NULL, 0},
                                              {NvSciBufImageAttrKey_Size, NULL, 0}};

        size_t slot_count = NvSciBufAttrListGetSlotCount(buf_attrs);
        for (size_t slot_index = 0; slot_index < slot_count; ++slot_index) {

            for (size_t i = 0; i < sizeof(keyVals) / sizeof(NvSciBufAttrKeyValuePair); ++i) {
                // NvSciError err = NvSciBufAttrListGetAttrs(buf_attrs, &keyVals[i], 1);
                NvSciError err = NvSciBufAttrListSlotGetAttrs(buf_attrs, slot_index, &keyVals[i], 1);
                if (NvSciError_Success != err) {
                    // printf("Failed (%x) to obtain buffer attribute: %s from slot %d.\n", err, hozon::netaos::nv::NvsUtility::GetKeyName(keyVals[i].key), static_cast<int>(slot_index));
                    return -1;
                }
            }
        }

        // uint32_t plane_count = *static_cast<const uint32_t*>(keyVals[0].value);
        // Take the width of y plane ans yuv pixel width
        sensor_info_mapping_[topic].resolution_width = *static_cast<const uint32_t*>(keyVals[2].value);
        // Take the height of y plane as yuv pixel heigth
        sensor_info_mapping_[topic].resolution_height = *static_cast<const uint32_t*>(keyVals[3].value);
        // sensor_info_mapping_[topic].yuv_buf_size = *static_cast<const uint64_t*>(keyVals[4].value);

        // TODO: calculate frame size.

        return encoder_map_[topic]->SetEncoderParam(hozon::netaos::codec::kEncoderParam_Nv_BufAttrs, static_cast<void*>(buf_attrs));
    }

    return -1;
}

int32_t CameraVenc::OnSetBufObj(std::string topic, NvSciBufObj buf_obj) {
    CAMV_INFO << "Set buf obj on topic: " << topic;
    if (encoder_map_.find(topic) != encoder_map_.end()) {
        return encoder_map_[topic]->SetEncoderParam(hozon::netaos::codec::kEncoderParam_Nv_BufObj, static_cast<void*>(buf_obj));
    }

    return -1;
}

void CameraVenc::OnH265Aailable(std::string topic, hozon::netaos::desay::Multicast_EncodedImage& encoded_image) {
    // Put h265 to sender.
    hozon::soc::CompressedImage h265_image;
    // h265_image.mutable_header().set_timestamp_sec(); // double
    h265_image.mutable_header()->set_seq(encoded_image.seq);  // uint32_t
    // h265_image.mutable_header().set_camera_timestamp(); // uint64_t
    // h265_image.mutable_header()->set_version(1);
    // h265_image.mutable_header().set_status(1); // StatusPb
    // h265_image.mutable_header().set_frame_id();
    h265_image.mutable_header()->mutable_sensor_stamp()->set_camera_stamp(encoded_image.sensor_time / 1000000.0);
    double pub_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    h265_image.mutable_header()->set_publish_stamp(pub_time);
    h265_image.set_format("h265");
    h265_image.set_length(encoded_image.data.size());
    h265_image.set_data(std::move(encoded_image.data));
    // h265_image.set_measurement_time();
    h265_image.set_frame_type(ConvertNvMediaFrameType(static_cast<NvMediaEncodeH26xFrameType>(encoded_image.frame_type)));
    h265_image.set_measurement_time(encoded_image.recv_time);

    if (!h265_sender_.Put(sensor_info_mapping_[topic].enc_topic, h265_image)) {
        CAMV_ERROR << topic << "  dds write error!";
    }

    if (write_file_) {
        auto file = file_map_.find(topic);
        if (file != file_map_.end()) {
            file->second->write(encoded_image.data.data(), encoded_image.data.size());
        } else {
            CAMV_ERROR << topic << " not register, write file error.";
        }
    }
}

hozon::netaos::codec::FrameType CameraVenc::ConvertNvMediaFrameType(NvMediaEncodeH26xFrameType nvmedia_type) {
    hozon::netaos::codec::FrameType frame_type = hozon::netaos::codec::kFrameType_None;
    switch (nvmedia_type) {
        /** P Frame */
        case NVMEDIA_ENCODE_H26X_FRAME_TYPE_P:
            frame_type = hozon::netaos::codec::kFrameType_P;
            break;
        /** B Frame */
        case NVMEDIA_ENCODE_H26X_FRAME_TYPE_B:
            frame_type = hozon::netaos::codec::kFrameType_B;
            break;
        /** I Frame */
        case NVMEDIA_ENCODE_H26X_FRAME_TYPE_I:
            frame_type = hozon::netaos::codec::kFrameType_I;
            break;
        /** IDR Frame */
        case NVMEDIA_ENCODE_H26X_FRAME_TYPE_IDR:
            frame_type = hozon::netaos::codec::kFrameType_I;
            break;
        /** @note This value is for internal use only. */
        case NVMEDIA_ENCODE_H26X_FRAME_TYPE_END:
            break;
        default:
            break;
    }

    return frame_type;
}

#endif

}  // namespace cameravenc
}  // namespace netaos
}  // namespace hozon