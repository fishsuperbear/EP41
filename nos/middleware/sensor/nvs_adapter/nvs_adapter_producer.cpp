#include "sensor/nvs_adapter/nvs_adapter_producer.h"

namespace hozon {
namespace netaos {
namespace nv { 

int32_t NvStreamAdapterProducer::Init(
        const std::string& ipc_channel, const std::string& producer_name, const uint32_t num_packets, const uint32_t num_consumers) {

    NvSciStreamBlock producer_link;
    NvSciError err;

    int32_t ret = _nvs_pool.Create(num_packets);
    if (ret < 0) {
        return -1;
    }

    ret = _nvs_image_producer.Create(_nvs_pool.block, producer_name);
    if (ret < 0) {
        return -1;
    }

    producer_link = _nvs_image_producer.block;
    if (num_consumers > 1) {
        ret = _nvs_multicast.Create(num_consumers);
        if (ret < 0) {
            return -1;
        }

        err = NvSciStreamBlockConnect(_nvs_image_producer.block, _nvs_multicast.block);
        if (NvSciError_Success != err) {
            NVS_LOG_CRITICAL << "Fail to connect multicast to producer, ret " << LogHexNvErr(err);
            return -1;
        }

        producer_link = _nvs_multicast.block;
    }

    _nvs_ipcs.resize(num_consumers);
    for (uint32_t i = 0; i < _nvs_ipcs.size(); ++i) {
        ret = _nvs_ipcs[i].Create(ipc_channel);
        if (ret < 0) {
            return -1;
        }

        err = NvSciStreamBlockConnect(producer_link, _nvs_ipcs[i].block);
        if (NvSciError_Success != err) {
            NVS_LOG_CRITICAL << "Fail to connect consumer " << i << " to producer, ret " << LogHexNvErr(err);
            return -1;
        }
    }

    NVS_LOG_INFO << "Succ to init NvStreamAdapterProducer";

    return 0;
}

}
}
}