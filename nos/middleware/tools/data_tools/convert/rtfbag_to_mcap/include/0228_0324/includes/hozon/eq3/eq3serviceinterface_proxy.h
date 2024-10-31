/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_EQ3_EQ3SERVICEINTERFACE_PROXY_H
#define HOZON_EQ3_EQ3SERVICEINTERFACE_PROXY_H

#include "ara/com/internal/proxy/proxy_adapter.h"
#include "ara/com/internal/proxy/event_adapter.h"
#include "ara/com/internal/proxy/field_adapter.h"
#include "ara/com/internal/proxy/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/eq3/eq3serviceinterface_common.h"
#include <string>

namespace hozon {
namespace eq3 {
namespace proxy {
namespace events {
    using Eq3VisDataEvent = ara::com::internal::proxy::event::EventAdapter<::hozon::eq3::Eq3VisDataType>;
    using Eq3PedestrianDataEvent = ara::com::internal::proxy::event::EventAdapter<::hozon::eq3::PedestrianInfos>;
    using Eq3RtdisDataEvent = ara::com::internal::proxy::event::EventAdapter<::hozon::eq3::RTDisInfos>;
    using Eq3RtsdisDataEvent = ara::com::internal::proxy::event::EventAdapter<::hozon::eq3::RTSDisInfos>;
    using Eq3VisObsMsgEvent = ara::com::internal::proxy::event::EventAdapter<::hozon::eq3::VisObsMsgsDataType>;
    static constexpr ara::com::internal::EntityId Eq3ServiceInterfaceEq3VisDataEventId = 48405U; //Eq3VisDataEvent_event_hash
    static constexpr ara::com::internal::EntityId Eq3ServiceInterfaceEq3PedestrianDataEventId = 10191U; //Eq3PedestrianDataEvent_event_hash
    static constexpr ara::com::internal::EntityId Eq3ServiceInterfaceEq3RtdisDataEventId = 34878U; //Eq3RtdisDataEvent_event_hash
    static constexpr ara::com::internal::EntityId Eq3ServiceInterfaceEq3RtsdisDataEventId = 7435U; //Eq3RtsdisDataEvent_event_hash
    static constexpr ara::com::internal::EntityId Eq3ServiceInterfaceEq3VisObsMsgEventId = 3675U; //Eq3VisObsMsgEvent_event_hash
}

namespace fields {
}

namespace methods {

} // namespace methods

class Eq3ServiceInterfaceProxy {
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

    virtual ~Eq3ServiceInterfaceProxy()
    {
        Eq3VisDataEvent.UnsetReceiveHandler();
        Eq3VisDataEvent.Unsubscribe();
        Eq3PedestrianDataEvent.UnsetReceiveHandler();
        Eq3PedestrianDataEvent.Unsubscribe();
        Eq3RtdisDataEvent.UnsetReceiveHandler();
        Eq3RtdisDataEvent.Unsubscribe();
        Eq3RtsdisDataEvent.UnsetReceiveHandler();
        Eq3RtsdisDataEvent.Unsubscribe();
        Eq3VisObsMsgEvent.UnsetReceiveHandler();
        Eq3VisObsMsgEvent.Unsubscribe();
    }

    explicit Eq3ServiceInterfaceProxy(const vrtf::vcc::api::types::HandleType &handle)
        : proxyAdapter(std::make_unique<ara::com::internal::proxy::ProxyAdapter>(::hozon::eq3::Eq3ServiceInterface::ServiceIdentifier, handle)),
          Eq3VisDataEvent(proxyAdapter->GetProxy(), events::Eq3ServiceInterfaceEq3VisDataEventId, proxyAdapter->GetHandle(), ::hozon::eq3::Eq3ServiceInterface::ServiceIdentifier),
          Eq3PedestrianDataEvent(proxyAdapter->GetProxy(), events::Eq3ServiceInterfaceEq3PedestrianDataEventId, proxyAdapter->GetHandle(), ::hozon::eq3::Eq3ServiceInterface::ServiceIdentifier),
          Eq3RtdisDataEvent(proxyAdapter->GetProxy(), events::Eq3ServiceInterfaceEq3RtdisDataEventId, proxyAdapter->GetHandle(), ::hozon::eq3::Eq3ServiceInterface::ServiceIdentifier),
          Eq3RtsdisDataEvent(proxyAdapter->GetProxy(), events::Eq3ServiceInterfaceEq3RtsdisDataEventId, proxyAdapter->GetHandle(), ::hozon::eq3::Eq3ServiceInterface::ServiceIdentifier),
          Eq3VisObsMsgEvent(proxyAdapter->GetProxy(), events::Eq3ServiceInterfaceEq3VisObsMsgEventId, proxyAdapter->GetHandle(), ::hozon::eq3::Eq3ServiceInterface::ServiceIdentifier){
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

    Eq3ServiceInterfaceProxy(const Eq3ServiceInterfaceProxy&) = delete;
    Eq3ServiceInterfaceProxy& operator=(const Eq3ServiceInterfaceProxy&) = delete;

    Eq3ServiceInterfaceProxy(Eq3ServiceInterfaceProxy&&) = default;
    Eq3ServiceInterfaceProxy& operator=(Eq3ServiceInterfaceProxy&&) = default;
    Eq3ServiceInterfaceProxy(ConstructionToken&& token) noexcept
        : proxyAdapter(token.GetProxyAdapter()),
          Eq3VisDataEvent(proxyAdapter->GetProxy(), events::Eq3ServiceInterfaceEq3VisDataEventId, proxyAdapter->GetHandle(), ::hozon::eq3::Eq3ServiceInterface::ServiceIdentifier),
          Eq3PedestrianDataEvent(proxyAdapter->GetProxy(), events::Eq3ServiceInterfaceEq3PedestrianDataEventId, proxyAdapter->GetHandle(), ::hozon::eq3::Eq3ServiceInterface::ServiceIdentifier),
          Eq3RtdisDataEvent(proxyAdapter->GetProxy(), events::Eq3ServiceInterfaceEq3RtdisDataEventId, proxyAdapter->GetHandle(), ::hozon::eq3::Eq3ServiceInterface::ServiceIdentifier),
          Eq3RtsdisDataEvent(proxyAdapter->GetProxy(), events::Eq3ServiceInterfaceEq3RtsdisDataEventId, proxyAdapter->GetHandle(), ::hozon::eq3::Eq3ServiceInterface::ServiceIdentifier),
          Eq3VisObsMsgEvent(proxyAdapter->GetProxy(), events::Eq3ServiceInterfaceEq3VisObsMsgEventId, proxyAdapter->GetHandle(), ::hozon::eq3::Eq3ServiceInterface::ServiceIdentifier){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        const vrtf::vcc::api::types::HandleType &handle)
    {
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> preProxyAdapter =
            std::make_unique<ara::com::internal::proxy::ProxyAdapter>(
               ::hozon::eq3::Eq3ServiceInterface::ServiceIdentifier, handle);
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
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::hozon::eq3::Eq3ServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType> handler,
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::hozon::eq3::Eq3ServiceInterface::ServiceIdentifier, specifier);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::hozon::eq3::Eq3ServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::hozon::eq3::Eq3ServiceInterface::ServiceIdentifier, specifier);
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
    events::Eq3VisDataEvent Eq3VisDataEvent;
    events::Eq3PedestrianDataEvent Eq3PedestrianDataEvent;
    events::Eq3RtdisDataEvent Eq3RtdisDataEvent;
    events::Eq3RtsdisDataEvent Eq3RtsdisDataEvent;
    events::Eq3VisObsMsgEvent Eq3VisObsMsgEvent;
};
} // namespace proxy
} // namespace eq3
} // namespace hozon

#endif // HOZON_EQ3_EQ3SERVICEINTERFACE_PROXY_H
