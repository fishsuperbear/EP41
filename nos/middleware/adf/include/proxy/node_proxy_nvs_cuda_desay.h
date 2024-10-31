#pragma once

#ifdef BUILD_FOR_ORIN

#include "adf/include/data_types/image/orin_image.h"
#include "adf/include/internal_log.h"
#include "adf/include/node_proto_register.h"
#include "adf/include/node_proxy.h"
#include "sensor/nvs_adapter/nvs_helper.h"
#include "sensor/nvs_consumer/CCudaConsumer.hpp"
#include "sensor/nvs_consumer/CIpcConsumerChannel.hpp"

#include "cm/include/method.h"
#include "idl/generated/sensor_reattachPubSubTypes.h"

namespace hozon {
namespace netaos {
namespace adf {

class NodeProxyNVSCUDADesay : public NodeProxyBase {
   public:
    explicit NodeProxyNVSCUDADesay(const NodeConfig::CommInstanceConfig& config);
    ~NodeProxyNVSCUDADesay();

    void OnDataReceive(void) override;
    void PauseReceive() override;
    void ResumeReceive() override;
    void Deinit() override;

   private:
    int32_t InitNVS(const std::string& ipc_channel);
    void NVSReadyCallback(std::shared_ptr<desay::DesayCUDAPacket> packet);
    void NVSReleaseBufferCB(bool need_free, void* dev_ptr);

    bool m_bquit = false;
    uint32_t _sensor_id = 0;
    uint32_t _channel_id = 0;
    std::shared_ptr<desay::CIpcConsumerChannel> _consumer;
    std::unique_ptr<hozon::netaos::cm::Client<sensor_reattach, sensor_reattach_resp>> _reattach_clint;
};

}  // namespace adf
}  // namespace netaos
}  // namespace hozon

#endif
