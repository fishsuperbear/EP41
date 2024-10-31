/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_ADAS_ADASSERVICEINTERFACE_PROXY_H
#define ARA_ADAS_ADASSERVICEINTERFACE_PROXY_H

#include "ara/com/internal/proxy/proxy_adapter.h"
#include "ara/com/internal/proxy/event_adapter.h"
#include "ara/com/internal/proxy/field_adapter.h"
#include "ara/com/internal/proxy/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "ara/adas/adasserviceinterface_common.h"
#include <string>

namespace ara {
namespace adas {
namespace proxy {
namespace events {
    using FLCFr01InfoEvent = ara::com::internal::proxy::event::EventAdapter<::ara::vehicle::FLCFr01Info>;
    using FLCFr02InfoEvent = ara::com::internal::proxy::event::EventAdapter<::ara::vehicle::FLCFr02Info>;
    using FLRFr01InfoEvent = ara::com::internal::proxy::event::EventAdapter<::ara::vehicle::FLRFr01Info>;
    using FLRFr02InfoEvent = ara::com::internal::proxy::event::EventAdapter<::ara::vehicle::FLRFr02Info>;
    using FLRFr03InfoEvent = ara::com::internal::proxy::event::EventAdapter<::ara::vehicle::FLRFr03Info>;
    using ApaInfoEvent = ara::com::internal::proxy::event::EventAdapter<::ara::vehicle::APAInfo>;
    static constexpr ara::com::internal::EntityId AdasServiceInterfaceFLCFr01InfoEventId = 60499U; //FLCFr01InfoEvent_event_hash
    static constexpr ara::com::internal::EntityId AdasServiceInterfaceFLCFr02InfoEventId = 17391U; //FLCFr02InfoEvent_event_hash
    static constexpr ara::com::internal::EntityId AdasServiceInterfaceFLRFr01InfoEventId = 30176U; //FLRFr01InfoEvent_event_hash
    static constexpr ara::com::internal::EntityId AdasServiceInterfaceFLRFr02InfoEventId = 2444U; //FLRFr02InfoEvent_event_hash
    static constexpr ara::com::internal::EntityId AdasServiceInterfaceFLRFr03InfoEventId = 38387U; //FLRFr03InfoEvent_event_hash
    static constexpr ara::com::internal::EntityId AdasServiceInterfaceApaInfoEventId = 52677U; //ApaInfoEvent_event_hash
}

namespace fields {
}

namespace methods {

} // namespace methods

class AdasServiceInterfaceProxy {
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

    virtual ~AdasServiceInterfaceProxy()
    {
        FLCFr01InfoEvent.UnsetReceiveHandler();
        FLCFr01InfoEvent.Unsubscribe();
        FLCFr02InfoEvent.UnsetReceiveHandler();
        FLCFr02InfoEvent.Unsubscribe();
        FLRFr01InfoEvent.UnsetReceiveHandler();
        FLRFr01InfoEvent.Unsubscribe();
        FLRFr02InfoEvent.UnsetReceiveHandler();
        FLRFr02InfoEvent.Unsubscribe();
        FLRFr03InfoEvent.UnsetReceiveHandler();
        FLRFr03InfoEvent.Unsubscribe();
        ApaInfoEvent.UnsetReceiveHandler();
        ApaInfoEvent.Unsubscribe();
    }

    explicit AdasServiceInterfaceProxy(const vrtf::vcc::api::types::HandleType &handle)
        : proxyAdapter(std::make_unique<ara::com::internal::proxy::ProxyAdapter>(::ara::adas::AdasServiceInterface::ServiceIdentifier, handle)),
          FLCFr01InfoEvent(proxyAdapter->GetProxy(), events::AdasServiceInterfaceFLCFr01InfoEventId, proxyAdapter->GetHandle(), ::ara::adas::AdasServiceInterface::ServiceIdentifier),
          FLCFr02InfoEvent(proxyAdapter->GetProxy(), events::AdasServiceInterfaceFLCFr02InfoEventId, proxyAdapter->GetHandle(), ::ara::adas::AdasServiceInterface::ServiceIdentifier),
          FLRFr01InfoEvent(proxyAdapter->GetProxy(), events::AdasServiceInterfaceFLRFr01InfoEventId, proxyAdapter->GetHandle(), ::ara::adas::AdasServiceInterface::ServiceIdentifier),
          FLRFr02InfoEvent(proxyAdapter->GetProxy(), events::AdasServiceInterfaceFLRFr02InfoEventId, proxyAdapter->GetHandle(), ::ara::adas::AdasServiceInterface::ServiceIdentifier),
          FLRFr03InfoEvent(proxyAdapter->GetProxy(), events::AdasServiceInterfaceFLRFr03InfoEventId, proxyAdapter->GetHandle(), ::ara::adas::AdasServiceInterface::ServiceIdentifier),
          ApaInfoEvent(proxyAdapter->GetProxy(), events::AdasServiceInterfaceApaInfoEventId, proxyAdapter->GetHandle(), ::ara::adas::AdasServiceInterface::ServiceIdentifier){
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

    AdasServiceInterfaceProxy(const AdasServiceInterfaceProxy&) = delete;
    AdasServiceInterfaceProxy& operator=(const AdasServiceInterfaceProxy&) = delete;

    AdasServiceInterfaceProxy(AdasServiceInterfaceProxy&&) = default;
    AdasServiceInterfaceProxy& operator=(AdasServiceInterfaceProxy&&) = default;
    AdasServiceInterfaceProxy(ConstructionToken&& token) noexcept
        : proxyAdapter(token.GetProxyAdapter()),
          FLCFr01InfoEvent(proxyAdapter->GetProxy(), events::AdasServiceInterfaceFLCFr01InfoEventId, proxyAdapter->GetHandle(), ::ara::adas::AdasServiceInterface::ServiceIdentifier),
          FLCFr02InfoEvent(proxyAdapter->GetProxy(), events::AdasServiceInterfaceFLCFr02InfoEventId, proxyAdapter->GetHandle(), ::ara::adas::AdasServiceInterface::ServiceIdentifier),
          FLRFr01InfoEvent(proxyAdapter->GetProxy(), events::AdasServiceInterfaceFLRFr01InfoEventId, proxyAdapter->GetHandle(), ::ara::adas::AdasServiceInterface::ServiceIdentifier),
          FLRFr02InfoEvent(proxyAdapter->GetProxy(), events::AdasServiceInterfaceFLRFr02InfoEventId, proxyAdapter->GetHandle(), ::ara::adas::AdasServiceInterface::ServiceIdentifier),
          FLRFr03InfoEvent(proxyAdapter->GetProxy(), events::AdasServiceInterfaceFLRFr03InfoEventId, proxyAdapter->GetHandle(), ::ara::adas::AdasServiceInterface::ServiceIdentifier),
          ApaInfoEvent(proxyAdapter->GetProxy(), events::AdasServiceInterfaceApaInfoEventId, proxyAdapter->GetHandle(), ::ara::adas::AdasServiceInterface::ServiceIdentifier){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        const vrtf::vcc::api::types::HandleType &handle)
    {
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> preProxyAdapter =
            std::make_unique<ara::com::internal::proxy::ProxyAdapter>(
               ::ara::adas::AdasServiceInterface::ServiceIdentifier, handle);
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
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::ara::adas::AdasServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType> handler,
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::ara::adas::AdasServiceInterface::ServiceIdentifier, specifier);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::ara::adas::AdasServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::ara::adas::AdasServiceInterface::ServiceIdentifier, specifier);
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
    events::FLCFr01InfoEvent FLCFr01InfoEvent;
    events::FLCFr02InfoEvent FLCFr02InfoEvent;
    events::FLRFr01InfoEvent FLRFr01InfoEvent;
    events::FLRFr02InfoEvent FLRFr02InfoEvent;
    events::FLRFr03InfoEvent FLRFr03InfoEvent;
    events::ApaInfoEvent ApaInfoEvent;
};
} // namespace proxy
} // namespace adas
} // namespace ara

#endif // ARA_ADAS_ADASSERVICEINTERFACE_PROXY_H
