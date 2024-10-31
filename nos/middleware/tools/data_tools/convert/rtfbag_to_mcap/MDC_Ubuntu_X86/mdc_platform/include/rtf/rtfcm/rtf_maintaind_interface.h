/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.
 * Description: This use to interface with maintaind.
 * Create: 2022-04-22
 */
#ifndef RTF_MAINTAIND_INTERFACE_H
#define RTF_MAINTAIND_INTERFACE_H
#include <future>
#include "ara/hwcommon/log/log.h"
#include "vrtf/vcc/api/proxy.h"
#include "vrtf/driver/dds/dds_driver_types.h"
#include "vrtf/driver/someip/someip_driver_types.h"
#include "vrtf/vcc/utils/thread_pool.h"
#include "vrtf/vcc/utils/event_controller.h"
#include "vrtf/vcc/utils/latency_analysis.h"
#include "ara/com/types.h"
namespace rtf {
namespace rtfcm {
namespace rtfmaintaind {
class RtfMaintaindInterface : public std::enable_shared_from_this<RtfMaintaindInterface> {
public:
    virtual ~RtfMaintaindInterface() = default;
    virtual void InitializeAll() = 0;
    virtual void RegisterNodePid(const vrtf::vcc::api::types::ServiceId& serviceId) = 0;
    virtual void RegisterNodePid(const vrtf::vcc::api::types::ServiceName& serviceName) = 0;
    virtual void RegisterEventInfoToMaintaind(
        const std::map<vrtf::vcc::api::types::DriverType, std::shared_ptr<vrtf::vcc::api::types::EventInfo>>& eventmap,
        const bool& isPub) = 0;
    virtual void UnregisterEventInfoToMaintaind(
        const std::map<vrtf::vcc::api::types::EntityId,
        std::map<vrtf::vcc::api::types::DriverType, std::shared_ptr<vrtf::vcc::api::types::EventInfo>>>& eventInfoList,
        const bool& isPub) noexcept = 0;
    virtual void RegisterMethodInfoToMaintaind(
        const std::map<vrtf::vcc::api::types::DriverType,
                       std::shared_ptr<vrtf::vcc::api::types::MethodInfo>>& methodmap,
        const bool& isPub) = 0;
    virtual void UnregisterMethodInfoToMaintaind(
        const std::map<vrtf::vcc::api::types::EntityId,
            std::map<vrtf::vcc::api::types::DriverType,
                     std::shared_ptr<vrtf::vcc::api::types::MethodInfo>>>& methodInfoList,
        const bool& isPub) noexcept = 0;
    virtual void RegisterFieldInfoToMaintaind(
        const std::map<vrtf::vcc::api::types::DriverType, std::shared_ptr<vrtf::vcc::api::types::FieldInfo>>& fieldmap,
        const bool& isPub) = 0;
    virtual void UnregisterFieldInfoToMaintaind(
        const std::map<vrtf::vcc::api::types::EntityId,
        std::map<vrtf::vcc::api::types::DriverType, std::shared_ptr<vrtf::vcc::api::types::FieldInfo>>>& fieldInfoList,
        const bool& isPub) noexcept = 0;
    virtual void RegisterLatencyMode(const vrtf::vcc::utils::SetLatencyModeHandler& handler,
        std::map<vrtf::vcc::api::types::DriverType, std::shared_ptr<vrtf::vcc::api::types::EventInfo>>& eventData,
        const bool& isPub) = 0;
    virtual void RegisterLatencyQuery(const vrtf::vcc::utils::LatencyQueryHandler& handler,
        const std::shared_ptr<vrtf::vcc::api::types::EventInfo>& eventInfo,
        const vrtf::vcc::api::types::DriverType& type) = 0;
    virtual void RegisterE2EInfoToMaintaind(
        const std::vector<vrtf::vcc::api::types::EventE2EConfigInfo>& eventE2EInfo) = 0;
    virtual vrtf::vcc::api::types::internal::VccListenerParams GetMaintaindClientListener() noexcept
    {
        return vrtf::vcc::api::types::internal::VccListenerParams();
    };
};
}
}
}
#endif
