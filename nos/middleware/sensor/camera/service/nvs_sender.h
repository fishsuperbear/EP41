#pragma once

#include "sensor/nvs_adapter/nvs_block_sipl_producer.h"
#include "sensor/nvs_adapter/nvs_block_multicast.h"
#include "sensor/nvs_adapter/nvs_block_pool.h"
#include "sensor/nvs_adapter/nvs_block_ipc_src.h"

namespace hozon {
namespace netaos {
namespace nv { 

class NVSSender {
public:
    int32_t Init(std::vector<std::string>& ipc_channels, 
            const std::string& producer_name, 
            const uint32_t num_packets, 
            const uint32_t num_consumers,
            INvSIPLClient::ConsumerDesc::OutputType output_type, 
            uint32_t sensor_id,
            INvSIPLCamera* sipl_camera);

    void Deinit();

    NVSBlockSIPLProducer _nvs_sipl_producer;
    NVSBlockMulticast _nvs_multicast;
    NVSBlockPool _nvs_pool;
    std::vector<NVSBlockIPCSrc> _nvs_ipcs;
};

}
}
}