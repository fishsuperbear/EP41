#include "sensor/nvs_adapter/nvs_adapter_consumer.h"

namespace hozon {
namespace netaos {
namespace nv { 

int32_t NvStreamAdapterConsumer::Init(const std::string& ipc_channel, const std::string& consumer_name, NVSBlockCUDAConsumer::PacketReadyCallback callback) {
    int32_t ret = _nvs_queue.Create(false);
    if (ret < 0) {
        return -1;
    }

    ret = _nvs_img_consumer.Create(_nvs_queue.block, consumer_name, callback);
    if (ret < 0) {
        return -2;
    }

    ret = _nvs_ipc_dst.Create(ipc_channel);
    if (ret < 0) {
        return -3;
    }

    NvSciError err = NvSciStreamBlockConnect(_nvs_ipc_dst.block, _nvs_img_consumer.block);
    if (NvSciError_Success != err) {
        NVS_LOG_CRITICAL << "Fail to connect consumer to producer, ret " << LogHexNvErr(err);
        return -4;
    }

    NVS_LOG_INFO << "Succ to init NvStreamAdapterConsumer";
    return 0;
}

void NvStreamAdapterConsumer::Deinit() {
    _nvs_img_consumer.Stop();
    _nvs_queue.Stop();
    _nvs_ipc_dst.Stop();
}

}
}
}