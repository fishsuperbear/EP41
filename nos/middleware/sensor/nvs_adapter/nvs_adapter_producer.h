#pragma once

#include "sensor/nvs_adapter/nvs_helper.h"
#include "sensor/nvs_adapter/nvs_logger.h"
#include "sensor/nvs_adapter/nvs_block_img_producer.h"
#include "sensor/nvs_adapter/nvs_block_multicast.h"
#include "sensor/nvs_adapter/nvs_block_pool.h"
#include "sensor/nvs_adapter/nvs_block_ipc_src.h"

namespace hozon {
namespace netaos {
namespace nv { 

class NvStreamAdapterProducer {
public:
    int32_t Init(const std::string& ipc_channel, const std::string& producer_name, const uint32_t num_packets, const uint32_t num_consumers);

    NVSBlockImgProducer _nvs_image_producer;
    NVSBlockMulticast _nvs_multicast;
    NVSBlockPool _nvs_pool;
    std::vector<NVSBlockIPCSrc> _nvs_ipcs;
};

}
}
}