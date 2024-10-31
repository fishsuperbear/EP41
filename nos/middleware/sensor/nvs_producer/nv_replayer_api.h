/**
 * @file nv_recorder_api.h
 * @author zax (maxiaotian@hozonauto.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-11
 * 
 * Copyright Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * 
 */

#ifndef MIDDLEWARE_SENSOR_NVS_PRODUCER_NV_REPLAYER_API_H_
#define MIDDLEWARE_SENSOR_NVS_PRODUCER_NV_REPLAYER_API_H_

#include <chrono>
#include <memory>
#include "codec/include/codec_def.h"

using hozon::netaos::codec::InputBufPtr;
using hozon::netaos::codec::PicInfos;

class NvReplayer {
   public:
    NvReplayer();
    ~NvReplayer();
    bool CreateProducerChannels(const PicInfos& pic_infos);
    bool Post(uint8_t sid, InputBufPtr buf);
    bool CloseAllChannels();

   private:
    class CRecoderMaster;
    std::unique_ptr<CRecoderMaster> impl_;
};

#endif  // MIDDLEWARE_SENSOR_NVS_PRODUCER_NV_REPLAYER_API_H_
