#pragma once

#include "sensor/nvs_adapter/nvs_helper.h"
#include "sensor/nvs_adapter/nvs_logger.h"
#include "sensor/nvs_adapter/nvs_block_queue.h"
#include "sensor/nvs_adapter/nvs_block_img_consumer.h"
#include "sensor/nvs_adapter/nvs_block_cuda_consumer.h"
#include "sensor/nvs_adapter/nvs_block_ipc_dst.h"

namespace hozon {
namespace netaos {
namespace nv { 

class NvStreamAdapterConsumer {
public:
    int32_t Init(const std::string& ipc_channel, const std::string& consumer_name, NVSBlockCUDAConsumer::PacketReadyCallback callback);
    void Deinit();

    NVSBlockQueue _nvs_queue;
    // NVSBlockImgConsumer _nvs_img_consumer;
    NVSBlockCUDAConsumer _nvs_img_consumer;
    NVSBlockIPCDst _nvs_ipc_dst;
};

}
}
}