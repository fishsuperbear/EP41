/******************************************************************************
 * Copyright (C) 2023 HOZON-AUTO Ltd. All rights reserved.
 * Licensed Hozon
 ******************************************************************************/
#ifndef CAMERA_VENC_H
#define CAMERA_VENC_H
#pragma once

#include <map>
#include <thread>
#include <unordered_map>
#include "camera_venc_config.h"
#include "codec/include/codec_def.h"
#include "codec/include/decoder_factory.h"
#include "codec/include/encoder_factory.h"
#include "h265_receiver.h"
#include "h265_sender.h"
#include "yuv_receiver.h"
#include "yuv_sender.h"
#include "proto/soc/sensor_image.pb.h"

#ifdef BUILD_FOR_ORIN
#include "include/multicast.h"
#include "nvmedia_common_encode.h"
#include "nvmedia_common_encode_decode.h"
#include "nvmedia_iep.h"
#include "nvmedia_iep_output_extradata.h"
#include "sensor/nvs_consumer/CEncManager.h"
#include "yuv_nvstream_receiver.h"
#include "yuv_nvstream_receiver_v2.h"
#endif

struct NvSciSignalerAttrList;

namespace hozon {
namespace netaos {
namespace cameravenc {

class CameraVenc {
   public:
    CameraVenc();
    ~CameraVenc();

    void Init();
    void Deinit();

    void Start();
    void Stop();

   private:
#ifndef BUILD_FOR_ORIN
    void ProcessTopic(std::string yuv_topic, std::string enc_topic);
#else
    int32_t OnYuvAvailable(std::string topic, hozon::netaos::nv::IEPPacket* packet, NvSciSyncFence& pre_fence, NvSciSyncFence& eof_fence);
    int32_t OnGetBufAttrs(std::string topic, NvSciBufAttrList& buf_attrs);
    int32_t OnGetWaiterAttrs(std::string topic, NvSciSyncAttrList& waiter_attrs);
    int32_t OnGetSignalerAttrs(std::string topic, NvSciSyncAttrList& signaler_attrs);
    int32_t OnSetSignalerObj(std::string topic, NvSciSyncObj signaler_obj);
    int32_t OnSetWaiterObj(std::string topic, NvSciSyncObj water_obj);
    int32_t OnSetBufAttrs(std::string topic, int32_t elem_type, NvSciBufAttrList buf_attrs);
    int32_t OnSetBufObj(std::string topic, NvSciBufObj buf_obj);
    void OnH265Aailable(std::string topic, hozon::netaos::desay::Multicast_EncodedImage& encoded_image);
    hozon::netaos::codec::FrameType ConvertNvMediaFrameType(NvMediaEncodeH26xFrameType nvmedia_type);
#endif
    // Receivers and senders.
    H265Receiver h265_receiver_;
    H265Sender h265_sender_;

#ifdef BUILD_FOR_ORIN
    // YuvNvStreamReceiver yuv_receiver_;
    YuvNvStreamReceiverV2 yuv_receiver_;
    SensorInfoMap sensor_info_mapping_;
#else
    YuvReceiver yuv_receiver_;
    std::map<std::string, SensorInfo> sensor_info_mapping_;
#endif
    // YuvSender yuv_sender_;

    // Yuv topic <-> h265 topic mapping.
    // std::map<std::string, std::string> yuv_h265_mapping_;
    std::map<std::string, std::string> h265_yuv_mapping_;

    // image info mapping.
    // struct ImageInfo {
    //     uint32_t yuv_frame_size = 0;
    //     uint32_t yuv_pix_fmt = 0;
    //     uint32_t yuv_height = 0;
    //     uint32_t yuv_width = 0;
    //     uint32_t yuv_buf_size = 0;
    //     std::string encoder_cfg_file;
    //     std::string enc_topic;
    // };

    // Yuv topic <-> encoder mapping
    std::map<std::string, std::unique_ptr<hozon::netaos::codec::Encoder>> encoder_map_;
    // H265 topic <-> decoder mapping
    std::map<std::string, std::unique_ptr<hozon::netaos::codec::Decoder>> decoder_map_;
    // file map
    std::unordered_map<std::string, std::unique_ptr<std::ofstream>> file_map_;

    // encoder threads
    std::vector<std::thread> work_threads_;

    // encoder / decoder mode.
    enum WorkMode { kWorkNone = 0, kWorkEncoder, kWorkDecoder };

    WorkMode work_mode_;
    bool write_file_;
    bool stopped_;
};

}  // namespace cameravenc
}  // namespace netaos
}  // namespace hozon
#endif