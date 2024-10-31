/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_VEHICLESERVICEINTERFACE_PROXY_H
#define ARA_VEHICLE_VEHICLESERVICEINTERFACE_PROXY_H

#include "ara/com/internal/proxy/proxy_adapter.h"
#include "ara/com/internal/proxy/event_adapter.h"
#include "ara/com/internal/proxy/field_adapter.h"
#include "ara/com/internal/proxy/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "ara/vehicle/vehicleserviceinterface_common.h"
#include <string>

namespace ara {
namespace vehicle {
namespace proxy {
namespace events {
    using VehicleDataEvent = ara::com::internal::proxy::event::EventAdapter<::ara::vehicle::VehicleInfo>;
    using FLCInfoEvent = ara::com::internal::proxy::event::EventAdapter<::ara::vehicle::FLCInfo>;
    using FLRInfoEvent = ara::com::internal::proxy::event::EventAdapter<::ara::vehicle::FLRInfo>;
    static constexpr ara::com::internal::EntityId VehicleServiceInterfaceVehicleDataEventId = 30140U; //VehicleDataEvent_event_hash
    static constexpr ara::com::internal::EntityId VehicleServiceInterfaceFLCInfoEventId = 41206U; //FLCInfoEvent_event_hash
    static constexpr ara::com::internal::EntityId VehicleServiceInterfaceFLRInfoEventId = 32763U; //FLRInfoEvent_event_hash
}

namespace fields {
}

namespace methods {

} // namespace methods

class VehicleServiceInterfaceProxy {
private:
    std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> proxyAdapter;
public:
    using HandleType = vrtf::vcc::api::types::HandleType;
    class ConstructionToken {
    public:
        explicit ConstructionToken(std::unique_ptr<ara::com::internal::proxy::ProxyAdapter>& proxy): ptr(std::move(proxy)){}
        explicit ConstructionToken(std::unique_ptr<ara::com::internal::proxy::ProxyAdapter>&& proxy): ptr(std::move(proxy)){}
        ConstructionToken(ConstructionToken&& other) : ptr(std::move(other.ptr)) {}
        ConstructionToken& operator=(ConstructionToken && other)
        {
            if (&other != this) {
                ptr = std::move(other.ptr);
            }
            return *this;
        }
        ConstructionToken(const ConstructionToken&) = delete;
        ConstructionToken& operator = (const ConstructionToken&) = delete;
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> GetProxyAdapter()
        {
            return std::move(ptr);
        }
    private:
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> ptr;
    };

    virtual ~VehicleServiceInterfaceProxy()
    {
        VehicleDataEvent.UnsetReceiveHandler();
        VehicleDataEvent.Unsubscribe();
        FLCInfoEvent.UnsetReceiveHandler();
        FLCInfoEvent.Unsubscribe();
        FLRInfoEvent.UnsetReceiveHandler();
        FLRInfoEvent.Unsubscribe();
    }

    explicit VehicleServiceInterfaceProxy(const vrtf::vcc::api::types::HandleType &handle)
        : proxyAdapter(std::make_unique<ara::com::internal::proxy::ProxyAdapter>(::ara::vehicle::VehicleServiceInterface::ServiceIdentifier, handle)),
          VehicleDataEvent(proxyAdapter->GetProxy(), events::VehicleServiceInterfaceVehicleDataEventId, proxyAdapter->GetHandle(), ::ara::vehicle::VehicleServiceInterface::ServiceIdentifier),
          FLCInfoEvent(proxyAdapter->GetProxy(), events::VehicleServiceInterfaceFLCInfoEventId, proxyAdapter->GetHandle(), ::ara::vehicle::VehicleServiceInterface::ServiceIdentifier),
          FLRInfoEvent(proxyAdapter->GetProxy(), events::VehicleServiceInterfaceFLRInfoEventId, proxyAdapter->GetHandle(), ::ara::vehicle::VehicleServiceInterface::ServiceIdentifier){
        }

    void ThrowError(const ara::core::Result<void>& result) const
    {
        if (!(result.HasValue())) {
#ifndef NOT_SUPPORT_EXCEPTIONS
            ara::core::ErrorCode errorcode(result.Error());
            throw ara::com::ComException(std::move(errorcode));
#else
            std::cerr << "Error: Not support exception, create proxy failed!"<< std::endl;
#endif
        }
    }

    VehicleServiceInterfaceProxy(const VehicleServiceInterfaceProxy&) = delete;
    VehicleServiceInterfaceProxy& operator=(const VehicleServiceInterfaceProxy&) = delete;

    VehicleServiceInterfaceProxy(VehicleServiceInterfaceProxy&&) = default;
    VehicleServiceInterfaceProxy& operator=(VehicleServiceInterfaceProxy&&) = default;
    VehicleServiceInterfaceProxy(ConstructionToken&& token) noexcept
        : proxyAdapter(token.GetProxyAdapter()),
          VehicleDataEvent(proxyAdapter->GetProxy(), events::VehicleServiceInterfaceVehicleDataEventId, proxyAdapter->GetHandle(), ::ara::vehicle::VehicleServiceInterface::ServiceIdentifier),
          FLCInfoEvent(proxyAdapter->GetProxy(), events::VehicleServiceInterfaceFLCInfoEventId, proxyAdapter->GetHandle(), ::ara::vehicle::VehicleServiceInterface::ServiceIdentifier),
          FLRInfoEvent(proxyAdapter->GetProxy(), events::VehicleServiceInterfaceFLRInfoEventId, proxyAdapter->GetHandle(), ::ara::vehicle::VehicleServiceInterface::ServiceIdentifier){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        const vrtf::vcc::api::types::HandleType &handle)
    {
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> preProxyAdapter =
            std::make_unique<ara::com::internal::proxy::ProxyAdapter>(
               ::ara::vehicle::VehicleServiceInterface::ServiceIdentifier, handle);
        const bool result = true;
        const ara::core::Result<void> initResult;
        do {
        } while(false);

        if (result) {
            ConstructionToken token(std::move(preProxyAdapter));
            return ara::core::Result<ConstructionToken>(std::move(token));
        } else {
            ConstructionToken token(std::move(preProxyAdapter));
            ara::core::Result<ConstructionToken> preResult(std::move(token));
            const ara::core::ErrorCode errorcode(initResult.Error());
            preResult.EmplaceError(errorcode);
            return preResult;
        }
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType>& handler,
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::ara::vehicle::VehicleServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType> handler,
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::ara::vehicle::VehicleServiceInterface::ServiceIdentifier, specifier);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::ara::vehicle::VehicleServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::ara::vehicle::VehicleServiceInterface::ServiceIdentifier, specifier);
    }

    static void StopFindService(const ara::com::FindServiceHandle& handle)
    {
        ara::com::internal::proxy::ProxyAdapter::StopFindService(handle);
    }

    HandleType GetHandle() const
    {
        return proxyAdapter->GetHandle();
    }
    bool SetEventThreadNumber(const std::uint16_t number, const std::uint16_t queueSize)
    {
        return proxyAdapter->SetEventThreadNumber(number, queueSize);
    }
    events::VehicleDataEvent VehicleDataEvent;
    events::FLCInfoEvent FLCInfoEvent;
    events::FLRInfoEvent FLRInfoEvent;
};
} // namespace proxy
} // namespace vehicle
} // namespace ara

#endif // ARA_VEHICLE_VEHICLESERVICEINTERFACE_PROXY_H
