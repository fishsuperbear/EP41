/**
 * * --------------------------------------------------------------------
 * * |                                                                  |
 * * |     _         _    _ _______ ____         _____ ____  __  __     |
 * * |    (_)   /\  | |  | |__   __/ __ \       / ____/ __ \|  \/  |    |
 * * |     _   /  \ | |  | |  | | | |  | |     | |   | |  | | \  / |    |
 * * |    | | / /\ \| |  | |  | | | |  | |     | |   | |  | | |\/| |    |
 * * |    | |/ ____ \ |__| |  | | | |__| |  _  | |___| |__| | |  | |    |
 * * |    |_/_/    \_\____/   |_|  \____/  (_)  \_____\____/|_|  |_|    |
 * * |                                                                  |
 * * --------------------------------------------------------------------
 *
 *  * Copyright @ 2020 iAuto (Shanghai) Co., Ltd.
 *  * All Rights Reserved.
 *  *
 *  * Redistribution and use in source and binary forms, with or without
 *  * modification, are NOT permitted except as agreed by
 *  * iAuto (Shanghai) Co., Ltd.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS,
 *  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 *
 * @file mcumaintainservice_proxy.h
 * @brief proxy.h
 * @date  
 *
 */
#ifndef HOZON_NETAOS_V1_MCUMAINTAINSERVICE_PROXY_H_
#define HOZON_NETAOS_V1_MCUMAINTAINSERVICE_PROXY_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include <memory>
#include "ara/core/instance_specifier.h"
#include "ara/com/types.h"
#include "mcumaintainservice_common.h"


namespace hozon {
namespace netaos {
namespace v1 {
inline namespace v0 {
namespace proxy{
namespace events{
namespace MCUmaintainService {
class MCUPlatState : public ara::com::ProxyMemberBase {
    public:
        MCUPlatState(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::ProxyInstance> instance, uint32_t idx);
        void SetReceiveHandler(ara::com::EventReceiveHandler handler);
        void SetSubscriptionStateChangeHandler(ara::com::SubscriptionStateChangeHandler handler);
        void Subscribe(std::size_t maxSampleCount);

        ara::com::SubscriptionState GetSubscriptionState() const;

        size_t GetFreeSampleCount() const noexcept;

        using F = std::function<void(ara::com::SamplePtr<::hozon::netaos::MCUDebugDataType const>)>;
        ara::core::Result<size_t> GetNewSamples(F&& f, size_t maxNumberOfSamples = std::numeric_limits<size_t>::max());

        void Unsubscribe();
        void UnsetSubscriptionStateChangeHandler();
        void UnsetReceiveHandler();
};
class MCUPlatCloudState : public ara::com::ProxyMemberBase {
    public:
        MCUPlatCloudState(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::ProxyInstance> instance, uint32_t idx);
        void SetReceiveHandler(ara::com::EventReceiveHandler handler);
        void SetSubscriptionStateChangeHandler(ara::com::SubscriptionStateChangeHandler handler);
        void Subscribe(std::size_t maxSampleCount);

        ara::com::SubscriptionState GetSubscriptionState() const;

        size_t GetFreeSampleCount() const noexcept;

        using F = std::function<void(ara::com::SamplePtr<::hozon::netaos::MCUCloudDataType const>)>;
        ara::core::Result<size_t> GetNewSamples(F&& f, size_t maxNumberOfSamples = std::numeric_limits<size_t>::max());

        void Unsubscribe();
        void UnsetSubscriptionStateChangeHandler();
        void UnsetReceiveHandler();
};
class MCUAdasState : public ara::com::ProxyMemberBase {
    public:
        MCUAdasState(const ara::core::String& name, const std::shared_ptr<ara::com::runtime::ProxyInstance> instance, uint32_t idx);
        void SetReceiveHandler(ara::com::EventReceiveHandler handler);
        void SetSubscriptionStateChangeHandler(ara::com::SubscriptionStateChangeHandler handler);
        void Subscribe(std::size_t maxSampleCount);

        ara::com::SubscriptionState GetSubscriptionState() const;

        size_t GetFreeSampleCount() const noexcept;

        using F = std::function<void(ara::com::SamplePtr<::hozon::netaos::DtState_ADAS const>)>;
        ara::core::Result<size_t> GetNewSamples(F&& f, size_t maxNumberOfSamples = std::numeric_limits<size_t>::max());

        void Unsubscribe();
        void UnsetSubscriptionStateChangeHandler();
        void UnsetReceiveHandler();
};
} // namespace MCUmaintainService
} // namespace events


class MCUmaintainServiceProxy{
    private:
        std::shared_ptr<ara::com::runtime::ProxyInstance> instance_;
    public:
        explicit MCUmaintainServiceProxy(const ara::com::HandleType& handle_type);
        ~MCUmaintainServiceProxy() = default;

        MCUmaintainServiceProxy(const MCUmaintainServiceProxy&) = delete;
        MCUmaintainServiceProxy& operator=(const MCUmaintainServiceProxy&) = delete;
        MCUmaintainServiceProxy(MCUmaintainServiceProxy&&) = default;
        MCUmaintainServiceProxy& operator=(MCUmaintainServiceProxy&&) = default;

        static ara::com::ServiceHandleContainer<ara::com::HandleType> FindService(ara::com::InstanceIdentifier instance = ara::com::InstanceIdentifier(ara::com::InstanceIdentifier::Any));

        static ara::com::ServiceHandleContainer<ara::com::HandleType> FindService(ara::core::InstanceSpecifier instance);

        static ara::com::FindServiceHandle StartFindService(ara::com::FindServiceHandler<ara::com::HandleType> handler,
                                                            ara::com::InstanceIdentifier instance = ara::com::InstanceIdentifier(ara::com::InstanceIdentifier::Any));

        static ara::com::FindServiceHandle StartFindService(ara::com::FindServiceHandler<ara::com::HandleType> handler, ara::core::InstanceSpecifier instance);

        static void StopFindService(ara::com::FindServiceHandle handle);

        ara::com::HandleType GetHandle() const;

    public:
        events::MCUmaintainService::MCUPlatState MCUPlatState;
        events::MCUmaintainService::MCUPlatCloudState MCUPlatCloudState;
        events::MCUmaintainService::MCUAdasState MCUAdasState;
};
} // namespace proxy
} // namespace v0
} // namespace v1
} // namespace netaos
} // namespace hozon


#endif // HOZON_NETAOS_V1_MCUMAINTAINSERVICE_PROXY_H_
/* EOF */