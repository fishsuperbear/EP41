/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_CONFIG_SERVER_CONFIGSERVERSERVICEINTERFACE_PROXY_H
#define MDC_CONFIG_SERVER_CONFIGSERVERSERVICEINTERFACE_PROXY_H

#include "ara/com/internal/proxy/proxy_adapter.h"
#include "ara/com/internal/proxy/event_adapter.h"
#include "ara/com/internal/proxy/field_adapter.h"
#include "ara/com/internal/proxy/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "mdc/config/server/configserverserviceinterface_common.h"
#include <string>

namespace mdc {
namespace config {
namespace server {
namespace proxy {
namespace events {
    using ParamUpdateEvent = ara::com::internal::proxy::event::EventAdapter<::mdc::config::server::ParamUpdateData>;
    using ServerNotifyEvent = ara::com::internal::proxy::event::EventAdapter<::mdc::config::server::ServerNotifyData>;
    static constexpr ara::com::internal::EntityId ConfigServerServiceInterfaceParamUpdateEventId = 35207U; //ParamUpdateEvent_event_hash
    static constexpr ara::com::internal::EntityId ConfigServerServiceInterfaceServerNotifyEventId = 27155U; //ServerNotifyEvent_event_hash
}

namespace fields {
}

namespace methods {
static constexpr ara::com::internal::EntityId ConfigServerServiceInterfaceAnswerAliveId = 30329U; //AnswerAlive_method_hash
static constexpr ara::com::internal::EntityId ConfigServerServiceInterfaceDelParamId = 56227U; //DelParam_method_hash
static constexpr ara::com::internal::EntityId ConfigServerServiceInterfaceGetMonitorClientsId = 42802U; //GetMonitorClients_method_hash
static constexpr ara::com::internal::EntityId ConfigServerServiceInterfaceGetParamId = 5505U; //GetParam_method_hash
static constexpr ara::com::internal::EntityId ConfigServerServiceInterfaceMonitorParamId = 3875U; //MonitorParam_method_hash
static constexpr ara::com::internal::EntityId ConfigServerServiceInterfaceSetParamId = 31959U; //SetParam_method_hash
static constexpr ara::com::internal::EntityId ConfigServerServiceInterfaceUnMonitorParamId = 36659U; //UnMonitorParam_method_hash
static constexpr ara::com::internal::EntityId ConfigServerServiceInterfaceInitClientId = 55067U; //InitClient_method_hash


class AnswerAlive {
public:
    using Output = mdc::config::server::methods::AnswerAlive::Output;

    AnswerAlive(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::String& clientName)
    {
        return method_(clientName);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::String> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::String> method_;
};

class DelParam {
public:
    using Output = mdc::config::server::methods::DelParam::Output;

    DelParam(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::String& clientName, const ::String& paramName)
    {
        return method_(clientName, paramName);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::String> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::String> method_;
};

class GetMonitorClients {
public:
    using Output = mdc::config::server::methods::GetMonitorClients::Output;

    GetMonitorClients(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::String& clientName, const ::String& paramName)
    {
        return method_(clientName, paramName);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::String> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::String> method_;
};

class GetParam {
public:
    using Output = mdc::config::server::methods::GetParam::Output;

    GetParam(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::String& clientName, const ::String& paramName)
    {
        return method_(clientName, paramName);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::String> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::String> method_;
};

class MonitorParam {
public:
    using Output = mdc::config::server::methods::MonitorParam::Output;

    MonitorParam(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::String& clientName, const ::String& paramName)
    {
        return method_(clientName, paramName);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::String> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::String> method_;
};

class SetParam {
public:
    using Output = mdc::config::server::methods::SetParam::Output;

    SetParam(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::String& clientName, const ::String& paramName, const ::String& paramValue, const ::UInt8& paramType, const ::UInt8& persistType)
    {
        return method_(clientName, paramName, paramValue, paramType, persistType);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::String, ::String, ::UInt8, ::UInt8> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::String, ::String, ::UInt8, ::UInt8> method_;
};

class UnMonitorParam {
public:
    using Output = mdc::config::server::methods::UnMonitorParam::Output;

    UnMonitorParam(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::String& clientName, const ::String& paramName)
    {
        return method_(clientName, paramName);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::String> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::String> method_;
};

class InitClient {
public:
    using Output = mdc::config::server::methods::InitClient::Output;

    InitClient(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::String& clientName)
    {
        return method_(clientName);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::String> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::String> method_;
};
} // namespace methods

class ConfigServerServiceInterfaceProxy {
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

    virtual ~ConfigServerServiceInterfaceProxy()
    {
        ParamUpdateEvent.UnsetReceiveHandler();
        ParamUpdateEvent.Unsubscribe();
        ServerNotifyEvent.UnsetReceiveHandler();
        ServerNotifyEvent.Unsubscribe();
    }

    explicit ConfigServerServiceInterfaceProxy(const vrtf::vcc::api::types::HandleType &handle)
        : proxyAdapter(std::make_unique<ara::com::internal::proxy::ProxyAdapter>(::mdc::config::server::ConfigServerServiceInterface::ServiceIdentifier, handle)),
          ParamUpdateEvent(proxyAdapter->GetProxy(), events::ConfigServerServiceInterfaceParamUpdateEventId, proxyAdapter->GetHandle(), ::mdc::config::server::ConfigServerServiceInterface::ServiceIdentifier),
          ServerNotifyEvent(proxyAdapter->GetProxy(), events::ConfigServerServiceInterfaceServerNotifyEventId, proxyAdapter->GetHandle(), ::mdc::config::server::ConfigServerServiceInterface::ServiceIdentifier),
          AnswerAlive(proxyAdapter->GetProxy(), methods::ConfigServerServiceInterfaceAnswerAliveId),
          DelParam(proxyAdapter->GetProxy(), methods::ConfigServerServiceInterfaceDelParamId),
          GetMonitorClients(proxyAdapter->GetProxy(), methods::ConfigServerServiceInterfaceGetMonitorClientsId),
          GetParam(proxyAdapter->GetProxy(), methods::ConfigServerServiceInterfaceGetParamId),
          MonitorParam(proxyAdapter->GetProxy(), methods::ConfigServerServiceInterfaceMonitorParamId),
          SetParam(proxyAdapter->GetProxy(), methods::ConfigServerServiceInterfaceSetParamId),
          UnMonitorParam(proxyAdapter->GetProxy(), methods::ConfigServerServiceInterfaceUnMonitorParamId),
          InitClient(proxyAdapter->GetProxy(), methods::ConfigServerServiceInterfaceInitClientId){
            ara::core::Result<void> resultAnswerAlive = proxyAdapter->InitializeMethod<methods::AnswerAlive::Output>(methods::ConfigServerServiceInterfaceAnswerAliveId);
            ThrowError(resultAnswerAlive);
            ara::core::Result<void> resultDelParam = proxyAdapter->InitializeMethod<methods::DelParam::Output>(methods::ConfigServerServiceInterfaceDelParamId);
            ThrowError(resultDelParam);
            ara::core::Result<void> resultGetMonitorClients = proxyAdapter->InitializeMethod<methods::GetMonitorClients::Output>(methods::ConfigServerServiceInterfaceGetMonitorClientsId);
            ThrowError(resultGetMonitorClients);
            ara::core::Result<void> resultGetParam = proxyAdapter->InitializeMethod<methods::GetParam::Output>(methods::ConfigServerServiceInterfaceGetParamId);
            ThrowError(resultGetParam);
            ara::core::Result<void> resultMonitorParam = proxyAdapter->InitializeMethod<methods::MonitorParam::Output>(methods::ConfigServerServiceInterfaceMonitorParamId);
            ThrowError(resultMonitorParam);
            ara::core::Result<void> resultSetParam = proxyAdapter->InitializeMethod<methods::SetParam::Output>(methods::ConfigServerServiceInterfaceSetParamId);
            ThrowError(resultSetParam);
            ara::core::Result<void> resultUnMonitorParam = proxyAdapter->InitializeMethod<methods::UnMonitorParam::Output>(methods::ConfigServerServiceInterfaceUnMonitorParamId);
            ThrowError(resultUnMonitorParam);
            ara::core::Result<void> resultInitClient = proxyAdapter->InitializeMethod<methods::InitClient::Output>(methods::ConfigServerServiceInterfaceInitClientId);
            ThrowError(resultInitClient);
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

    ConfigServerServiceInterfaceProxy(const ConfigServerServiceInterfaceProxy&) = delete;
    ConfigServerServiceInterfaceProxy& operator=(const ConfigServerServiceInterfaceProxy&) = delete;

    ConfigServerServiceInterfaceProxy(ConfigServerServiceInterfaceProxy&&) = default;
    ConfigServerServiceInterfaceProxy& operator=(ConfigServerServiceInterfaceProxy&&) = default;
    ConfigServerServiceInterfaceProxy(ConstructionToken&& token) noexcept
        : proxyAdapter(token.GetProxyAdapter()),
          ParamUpdateEvent(proxyAdapter->GetProxy(), events::ConfigServerServiceInterfaceParamUpdateEventId, proxyAdapter->GetHandle(), ::mdc::config::server::ConfigServerServiceInterface::ServiceIdentifier),
          ServerNotifyEvent(proxyAdapter->GetProxy(), events::ConfigServerServiceInterfaceServerNotifyEventId, proxyAdapter->GetHandle(), ::mdc::config::server::ConfigServerServiceInterface::ServiceIdentifier),
          AnswerAlive(proxyAdapter->GetProxy(), methods::ConfigServerServiceInterfaceAnswerAliveId),
          DelParam(proxyAdapter->GetProxy(), methods::ConfigServerServiceInterfaceDelParamId),
          GetMonitorClients(proxyAdapter->GetProxy(), methods::ConfigServerServiceInterfaceGetMonitorClientsId),
          GetParam(proxyAdapter->GetProxy(), methods::ConfigServerServiceInterfaceGetParamId),
          MonitorParam(proxyAdapter->GetProxy(), methods::ConfigServerServiceInterfaceMonitorParamId),
          SetParam(proxyAdapter->GetProxy(), methods::ConfigServerServiceInterfaceSetParamId),
          UnMonitorParam(proxyAdapter->GetProxy(), methods::ConfigServerServiceInterfaceUnMonitorParamId),
          InitClient(proxyAdapter->GetProxy(), methods::ConfigServerServiceInterfaceInitClientId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        const vrtf::vcc::api::types::HandleType &handle)
    {
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> preProxyAdapter =
            std::make_unique<ara::com::internal::proxy::ProxyAdapter>(
               ::mdc::config::server::ConfigServerServiceInterface::ServiceIdentifier, handle);
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            const methods::AnswerAlive AnswerAlive(preProxyAdapter->GetProxy(), methods::ConfigServerServiceInterfaceAnswerAliveId);
            initResult = preProxyAdapter->InitializeMethod<methods::AnswerAlive::Output>(methods::ConfigServerServiceInterfaceAnswerAliveId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::DelParam DelParam(preProxyAdapter->GetProxy(), methods::ConfigServerServiceInterfaceDelParamId);
            initResult = preProxyAdapter->InitializeMethod<methods::DelParam::Output>(methods::ConfigServerServiceInterfaceDelParamId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::GetMonitorClients GetMonitorClients(preProxyAdapter->GetProxy(), methods::ConfigServerServiceInterfaceGetMonitorClientsId);
            initResult = preProxyAdapter->InitializeMethod<methods::GetMonitorClients::Output>(methods::ConfigServerServiceInterfaceGetMonitorClientsId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::GetParam GetParam(preProxyAdapter->GetProxy(), methods::ConfigServerServiceInterfaceGetParamId);
            initResult = preProxyAdapter->InitializeMethod<methods::GetParam::Output>(methods::ConfigServerServiceInterfaceGetParamId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::MonitorParam MonitorParam(preProxyAdapter->GetProxy(), methods::ConfigServerServiceInterfaceMonitorParamId);
            initResult = preProxyAdapter->InitializeMethod<methods::MonitorParam::Output>(methods::ConfigServerServiceInterfaceMonitorParamId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::SetParam SetParam(preProxyAdapter->GetProxy(), methods::ConfigServerServiceInterfaceSetParamId);
            initResult = preProxyAdapter->InitializeMethod<methods::SetParam::Output>(methods::ConfigServerServiceInterfaceSetParamId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::UnMonitorParam UnMonitorParam(preProxyAdapter->GetProxy(), methods::ConfigServerServiceInterfaceUnMonitorParamId);
            initResult = preProxyAdapter->InitializeMethod<methods::UnMonitorParam::Output>(methods::ConfigServerServiceInterfaceUnMonitorParamId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::InitClient InitClient(preProxyAdapter->GetProxy(), methods::ConfigServerServiceInterfaceInitClientId);
            initResult = preProxyAdapter->InitializeMethod<methods::InitClient::Output>(methods::ConfigServerServiceInterfaceInitClientId);
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
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::mdc::config::server::ConfigServerServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType> handler,
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::mdc::config::server::ConfigServerServiceInterface::ServiceIdentifier, specifier);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::mdc::config::server::ConfigServerServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::mdc::config::server::ConfigServerServiceInterface::ServiceIdentifier, specifier);
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
    events::ParamUpdateEvent ParamUpdateEvent;
    events::ServerNotifyEvent ServerNotifyEvent;
    methods::AnswerAlive AnswerAlive;
    methods::DelParam DelParam;
    methods::GetMonitorClients GetMonitorClients;
    methods::GetParam GetParam;
    methods::MonitorParam MonitorParam;
    methods::SetParam SetParam;
    methods::UnMonitorParam UnMonitorParam;
    methods::InitClient InitClient;
};
} // namespace proxy
} // namespace server
} // namespace config
} // namespace mdc

#endif // MDC_CONFIG_SERVER_CONFIGSERVERSERVICEINTERFACE_PROXY_H
