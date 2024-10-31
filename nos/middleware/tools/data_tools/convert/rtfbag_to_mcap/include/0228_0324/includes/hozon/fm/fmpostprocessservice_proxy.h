/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_FM_FMPOSTPROCESSSERVICE_PROXY_H
#define HOZON_FM_FMPOSTPROCESSSERVICE_PROXY_H

#include "ara/com/internal/proxy/proxy_adapter.h"
#include "ara/com/internal/proxy/event_adapter.h"
#include "ara/com/internal/proxy/field_adapter.h"
#include "ara/com/internal/proxy/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/fm/fmpostprocessservice_common.h"
#include <string>

namespace hozon {
namespace fm {
namespace proxy {
namespace events {
    using FaultPostProcessEvent = ara::com::internal::proxy::event::EventAdapter<::hozon::fm::FaultClusterData>;
    static constexpr ara::com::internal::EntityId FmPostProcessServiceFaultPostProcessEventId = 26600U; //FaultPostProcessEvent_event_hash
}

namespace fields {
}

namespace methods {
static constexpr ara::com::internal::EntityId FmPostProcessServiceRegistPostProcessFaultId = 34209U; //RegistPostProcessFault_method_hash


class RegistPostProcessFault {
public:
    using Output = void;

    RegistPostProcessFault(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    void operator()(const ::String& appName, const ::StringVector& clusterList)
    {
        method_(appName, clusterList);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::StringVector> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::StringVector> method_;
};
} // namespace methods

class FmPostProcessServiceProxy {
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

    virtual ~FmPostProcessServiceProxy()
    {
        FaultPostProcessEvent.UnsetReceiveHandler();
        FaultPostProcessEvent.Unsubscribe();
    }

    explicit FmPostProcessServiceProxy(const vrtf::vcc::api::types::HandleType &handle)
        : proxyAdapter(std::make_unique<ara::com::internal::proxy::ProxyAdapter>(::hozon::fm::FmPostProcessService::ServiceIdentifier, handle)),
          FaultPostProcessEvent(proxyAdapter->GetProxy(), events::FmPostProcessServiceFaultPostProcessEventId, proxyAdapter->GetHandle(), ::hozon::fm::FmPostProcessService::ServiceIdentifier),
          RegistPostProcessFault(proxyAdapter->GetProxy(), methods::FmPostProcessServiceRegistPostProcessFaultId){
            ara::core::Result<void> resultRegistPostProcessFault = proxyAdapter->InitializeMethod<methods::RegistPostProcessFault::Output>(methods::FmPostProcessServiceRegistPostProcessFaultId);
            ThrowError(resultRegistPostProcessFault);
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

    FmPostProcessServiceProxy(const FmPostProcessServiceProxy&) = delete;
    FmPostProcessServiceProxy& operator=(const FmPostProcessServiceProxy&) = delete;

    FmPostProcessServiceProxy(FmPostProcessServiceProxy&&) = default;
    FmPostProcessServiceProxy& operator=(FmPostProcessServiceProxy&&) = default;
    FmPostProcessServiceProxy(ConstructionToken&& token) noexcept
        : proxyAdapter(token.GetProxyAdapter()),
          FaultPostProcessEvent(proxyAdapter->GetProxy(), events::FmPostProcessServiceFaultPostProcessEventId, proxyAdapter->GetHandle(), ::hozon::fm::FmPostProcessService::ServiceIdentifier),
          RegistPostProcessFault(proxyAdapter->GetProxy(), methods::FmPostProcessServiceRegistPostProcessFaultId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        const vrtf::vcc::api::types::HandleType &handle)
    {
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> preProxyAdapter =
            std::make_unique<ara::com::internal::proxy::ProxyAdapter>(
               ::hozon::fm::FmPostProcessService::ServiceIdentifier, handle);
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            const methods::RegistPostProcessFault RegistPostProcessFault(preProxyAdapter->GetProxy(), methods::FmPostProcessServiceRegistPostProcessFaultId);
            initResult = preProxyAdapter->InitializeMethod<methods::RegistPostProcessFault::Output>(methods::FmPostProcessServiceRegistPostProcessFaultId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
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
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::hozon::fm::FmPostProcessService::ServiceIdentifier, instance);
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType> handler,
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::hozon::fm::FmPostProcessService::ServiceIdentifier, specifier);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::hozon::fm::FmPostProcessService::ServiceIdentifier, instance);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::hozon::fm::FmPostProcessService::ServiceIdentifier, specifier);
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
    events::FaultPostProcessEvent FaultPostProcessEvent;
    methods::RegistPostProcessFault RegistPostProcessFault;
};
} // namespace proxy
} // namespace fm
} // namespace hozon

#endif // HOZON_FM_FMPOSTPROCESSSERVICE_PROXY_H
