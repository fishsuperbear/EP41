/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_SM_PLATFORMSTATEAGENT_PROXY_H
#define ARA_SM_PLATFORMSTATEAGENT_PROXY_H

#include "ara/com/internal/proxy/proxy_adapter.h"
#include "ara/com/internal/proxy/event_adapter.h"
#include "ara/com/internal/proxy/field_adapter.h"
#include "ara/com/internal/proxy/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "ara/sm/platformstateagent_common.h"
#include <string>

namespace ara {
namespace sm {
namespace proxy {
namespace events {
    using PlatformStateEvent = ara::com::internal::proxy::event::EventAdapter<::ara::sm::PlatformStateMsg>;
    static constexpr ara::com::internal::EntityId PlatformStateAgentPlatformStateEventId = 43376U; //PlatformStateEvent_event_hash
}

namespace fields {
}

namespace methods {
static constexpr ara::com::internal::EntityId PlatformStateAgentQueryPlatformStateId = 44471U; //QueryPlatformState_method_hash
static constexpr ara::com::internal::EntityId PlatformStateAgentRequestPlatformStateId = 37348U; //RequestPlatformState_method_hash


class QueryPlatformState {
public:
    using Output = ara::sm::methods::QueryPlatformState::Output;

    QueryPlatformState(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()()
    {
        return method_();
    }

    ara::com::internal::proxy::method::MethodAdapter<Output> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output> method_;
};

class RequestPlatformState {
public:
    using Output = ara::sm::methods::RequestPlatformState::Output;

    RequestPlatformState(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::uint8_t& platformStateIn, const ::Uint32Vector& data)
    {
        return method_(platformStateIn, data);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::uint8_t, ::Uint32Vector> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::uint8_t, ::Uint32Vector> method_;
};
} // namespace methods

class PlatformStateAgentProxy {
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

    virtual ~PlatformStateAgentProxy()
    {
        PlatformStateEvent.UnsetReceiveHandler();
        PlatformStateEvent.Unsubscribe();
    }

    explicit PlatformStateAgentProxy(const vrtf::vcc::api::types::HandleType &handle)
        : proxyAdapter(std::make_unique<ara::com::internal::proxy::ProxyAdapter>(::ara::sm::PlatformStateAgent::ServiceIdentifier, handle)),
          PlatformStateEvent(proxyAdapter->GetProxy(), events::PlatformStateAgentPlatformStateEventId, proxyAdapter->GetHandle(), ::ara::sm::PlatformStateAgent::ServiceIdentifier),
          QueryPlatformState(proxyAdapter->GetProxy(), methods::PlatformStateAgentQueryPlatformStateId),
          RequestPlatformState(proxyAdapter->GetProxy(), methods::PlatformStateAgentRequestPlatformStateId){
            ara::core::Result<void> resultQueryPlatformState = proxyAdapter->InitializeMethod<methods::QueryPlatformState::Output>(methods::PlatformStateAgentQueryPlatformStateId);
            ThrowError(resultQueryPlatformState);
            ara::core::Result<void> resultRequestPlatformState = proxyAdapter->InitializeMethod<methods::RequestPlatformState::Output>(methods::PlatformStateAgentRequestPlatformStateId);
            ThrowError(resultRequestPlatformState);
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

    PlatformStateAgentProxy(const PlatformStateAgentProxy&) = delete;
    PlatformStateAgentProxy& operator=(const PlatformStateAgentProxy&) = delete;

    PlatformStateAgentProxy(PlatformStateAgentProxy&&) = default;
    PlatformStateAgentProxy& operator=(PlatformStateAgentProxy&&) = default;
    PlatformStateAgentProxy(ConstructionToken&& token) noexcept
        : proxyAdapter(token.GetProxyAdapter()),
          PlatformStateEvent(proxyAdapter->GetProxy(), events::PlatformStateAgentPlatformStateEventId, proxyAdapter->GetHandle(), ::ara::sm::PlatformStateAgent::ServiceIdentifier),
          QueryPlatformState(proxyAdapter->GetProxy(), methods::PlatformStateAgentQueryPlatformStateId),
          RequestPlatformState(proxyAdapter->GetProxy(), methods::PlatformStateAgentRequestPlatformStateId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        const vrtf::vcc::api::types::HandleType &handle)
    {
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> preProxyAdapter =
            std::make_unique<ara::com::internal::proxy::ProxyAdapter>(
               ::ara::sm::PlatformStateAgent::ServiceIdentifier, handle);
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            const methods::QueryPlatformState QueryPlatformState(preProxyAdapter->GetProxy(), methods::PlatformStateAgentQueryPlatformStateId);
            initResult = preProxyAdapter->InitializeMethod<methods::QueryPlatformState::Output>(methods::PlatformStateAgentQueryPlatformStateId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::RequestPlatformState RequestPlatformState(preProxyAdapter->GetProxy(), methods::PlatformStateAgentRequestPlatformStateId);
            initResult = preProxyAdapter->InitializeMethod<methods::RequestPlatformState::Output>(methods::PlatformStateAgentRequestPlatformStateId);
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
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::ara::sm::PlatformStateAgent::ServiceIdentifier, instance);
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType> handler,
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::ara::sm::PlatformStateAgent::ServiceIdentifier, specifier);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::ara::sm::PlatformStateAgent::ServiceIdentifier, instance);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::ara::sm::PlatformStateAgent::ServiceIdentifier, specifier);
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
    events::PlatformStateEvent PlatformStateEvent;
    methods::QueryPlatformState QueryPlatformState;
    methods::RequestPlatformState RequestPlatformState;
};
} // namespace proxy
} // namespace sm
} // namespace ara

#endif // ARA_SM_PLATFORMSTATEAGENT_PROXY_H
