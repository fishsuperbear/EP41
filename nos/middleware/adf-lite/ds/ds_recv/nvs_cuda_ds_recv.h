#pragma once

#ifdef BUILD_FOR_ORIN

#include "adf-lite/ds/ds_recv/ds_recv.h"
#include "adf-lite/include/executor.h"
#include "cm/include/proxy.h"
#include "adf-lite/include/data_types/image/orin_image.h"

#include "sensor/nvs_consumer/CIpcConsumerChannel.hpp"
#include "sensor/nvs_consumer/CCudaConsumer.hpp"
#include "sensor/nvs_adapter/nvs_block_cuda_consumer.h"
#include "sensor/nvs_adapter/nvs_helper.h"

#include "idl/generated/sensor_reattachPubSubTypes.h"
#include "cm/include/method.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

class NvsCudaDesayDsRecv : public DsRecv {
public:
    explicit NvsCudaDesayDsRecv(const DSConfig::DataSource& config);
    ~NvsCudaDesayDsRecv();

    void OnDataReceive(void);
    void PauseReceive();
    void ResumeReceive();
    void Deinit();

private:
    int32_t InitNVS(const std::string& ipc_channel);
    void NVSReadyCallback(std::shared_ptr<desay::DesayCUDAPacket> packet);
    void NVSReleaseBufferCB(bool need_free, void* dev_ptr);

    std::shared_ptr<desay::CIpcConsumerChannel> _consumer;
    Writer _writer;
    bool m_bquit = false;
    uint32_t _sensor_id = 0;
    uint32_t _channel_id = 0;
    std::unique_ptr<hozon::netaos::cm::Client<sensor_reattach, sensor_reattach_resp>> _reattach_clint;
};


}
}
}

#endif
