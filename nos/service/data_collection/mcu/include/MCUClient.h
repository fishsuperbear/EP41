#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <sys/stat.h>
#include <ara/core/initialization.h>
#include "hozon/netaos/v1/mcucanmsgservice_proxy.h"
#include "hozon/netaos/v1/mcumaintainservice_proxy.h"
#include "hozon/netaos/v1/triggeridservice_skeleton.h"
#include "cm/include/skeleton.h"

namespace hozon {
namespace netaos {
namespace dc {

using hozon::netaos::v1::proxy::McuCANMsgServiceProxy;
using hozon::netaos::v1::proxy::MCUmaintainServiceProxy;
using hozon::netaos::v1::skeleton::TriggerIDServiceSkeleton;
using hozon::netaos::v1::skeleton::methods::TriggerIDService::MCUCloudTrigger;

class MCUClient {
   public:
    bool Init();
    void Deinit();
    static bool TriggerDataCollection(uint8_t triggerId);

   private:
    std::mutex m_proxy_mutex;
    std::shared_ptr<McuCANMsgServiceProxy> m_can_msg_service_proxy{nullptr};
    ara::com::FindServiceHandle m_find_can_msg_service_handler;
    std::shared_ptr<MCUmaintainServiceProxy> m_maintain_service_proxy{nullptr};
    ara::com::FindServiceHandle m_find_maintain_service_handler;

    void HandleCanMsgServiceSOMEIPData();
    void HandleMaintainServiceSOMEIPData();

    class TriggerIDServiceInstance : public TriggerIDServiceSkeleton {
       public:
        TriggerIDServiceInstance();
        bool Init();
        void DeInit();

       private:
        std::mutex m_skeleton_mutex;
        ara::core::Future<MCUCloudTrigger::Output> MCUCloudTrigger(const std::uint8_t& CloudTriggerID) override;
    };

    std::shared_ptr<TriggerIDServiceInstance> m_trigger_id_service_instance{nullptr};
    static std::unique_ptr<hozon::netaos::cm::Skeleton> skeleton;
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon
