#include "sensor/camera/service/nvs_sender.h"

namespace hozon {
namespace netaos {
namespace nv { 

int32_t NVSSender::Init(std::vector<std::string>& ipc_channels, 
            const std::string& producer_name, 
            const uint32_t num_packets, 
            const uint32_t num_consumers,
            INvSIPLClient::ConsumerDesc::OutputType output_type, 
            uint32_t sensor_id,
            INvSIPLCamera* sipl_camera) {
    NvSciStreamBlock producer_link;
    NvSciError err;

    int32_t ret = _nvs_pool.Create(num_packets);
    if (ret < 0) {
        return -1;
    }

    ret = _nvs_sipl_producer.Create(_nvs_pool.block, producer_name, output_type, sensor_id, sipl_camera);
    if (ret < 0) {
        return -1;
    }

    producer_link = _nvs_sipl_producer.block;
    if (num_consumers > 1) {
        ret = _nvs_multicast.Create(num_consumers);
        if (ret < 0) {
            return -1;
        }

        err = NvSciStreamBlockConnect(_nvs_sipl_producer.block, _nvs_multicast.block);
        if (NvSciError_Success != err) {
            NVS_LOG_CRITICAL << "Fail to connect multicast to producer, ret " << LogHexNvErr(err);
            return -1;
        }

        producer_link = _nvs_multicast.block;
    }

    _nvs_ipcs.resize(num_consumers);
    for (uint32_t i = 0; i < _nvs_ipcs.size(); ++i) {
        ret = _nvs_ipcs[i].Create(ipc_channels.at(i));
        if (ret < 0) {
            return -1;
        }
        // NVS_LOG_INFO << "Listen on ipc channel: " << ipc_channels.at(i);

        err = NvSciStreamBlockConnect(producer_link, _nvs_ipcs[i].block);
        if (NvSciError_Success != err) {
            NVS_LOG_CRITICAL << "Fail to connect consumer " << i << " to producer, ret " << LogHexNvErr(err);
            return -1;
        }
        NVS_LOG_INFO << "Connect ipc " << ipc_channels.at(i);
    }

    NVS_LOG_INFO << "Succ to init NvStreamAdapterProducer";

    return 0;
}

void NVSSender::Deinit() {
    _nvs_sipl_producer.Stop();
    _nvs_pool.Stop();
    
    for (auto& ipc : _nvs_ipcs) {
        ipc.Stop();
    }
}

}
}
}