/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CONFIG_SERVER_HZCONFIGSERVERSOMEIPSERVICEINTERFACE_PROXY_H
#define HOZON_CONFIG_SERVER_HZCONFIGSERVERSOMEIPSERVICEINTERFACE_PROXY_H

#include "ara/com/internal/proxy/proxy_adapter.h"
#include "ara/com/internal/proxy/event_adapter.h"
#include "ara/com/internal/proxy/field_adapter.h"
#include "ara/com/internal/proxy/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/config/server/hzconfigserversomeipserviceinterface_common.h"
#include <string>

namespace hozon {
namespace config {
namespace server {
namespace proxy {
namespace events {
    using HzParamUpdateEvent = ara::com::internal::proxy::event::EventAdapter<::hozon::config::server::HzParamUpdateData>;
    using HzServerNotifyEvent = ara::com::internal::proxy::event::EventAdapter<::hozon::config::server::HzServerNotifyData>;
    using VehicleCfgUpdateToMcuEvent = ara::com::internal::proxy::event::EventAdapter<::hozon::config::server::struct_config_array>;
    static constexpr ara::com::internal::EntityId HzConfigServerSomeipServiceInterfaceHzParamUpdateEventId = 29525U; //HzParamUpdateEvent_event_hash
    static constexpr ara::com::internal::EntityId HzConfigServerSomeipServiceInterfaceHzServerNotifyEventId = 3223U; //HzServerNotifyEvent_event_hash
    static constexpr ara::com::internal::EntityId HzConfigServerSomeipServiceInterfaceVehicleCfgUpdateToMcuEventId = 5435U; //VehicleCfgUpdateToMcuEvent_event_hash
}

namespace fields {
}

namespace methods {
static constexpr ara::com::internal::EntityId HzConfigServerSomeipServiceInterfaceHzAnswerAliveId = 17572U; //HzAnswerAlive_method_hash
static constexpr ara::com::internal::EntityId HzConfigServerSomeipServiceInterfaceHzDelParamId = 1648U; //HzDelParam_method_hash
static constexpr ara::com::internal::EntityId HzConfigServerSomeipServiceInterfaceHzGetMonitorClientsId = 2527U; //HzGetMonitorClients_method_hash
static constexpr ara::com::internal::EntityId HzConfigServerSomeipServiceInterfaceHzGetParamId = 27336U; //HzGetParam_method_hash
static constexpr ara::com::internal::EntityId HzConfigServerSomeipServiceInterfaceHzMonitorParamId = 62445U; //HzMonitorParam_method_hash
static constexpr ara::com::internal::EntityId HzConfigServerSomeipServiceInterfaceHzSetParamId = 48345U; //HzSetParam_method_hash
static constexpr ara::com::internal::EntityId HzConfigServerSomeipServiceInterfaceHzUnMonitorParamId = 44651U; //HzUnMonitorParam_method_hash
static constexpr ara::com::internal::EntityId HzConfigServerSomeipServiceInterfaceVehicleCfgUpdateResFromMcuId = 8777U; //VehicleCfgUpdateResFromMcu_method_hash
static constexpr ara::com::internal::EntityId HzConfigServerSomeipServiceInterfaceHzGetVehicleCfgParamId = 16330U; //HzGetVehicleCfgParam_method_hash
static constexpr ara::com::internal::EntityId HzConfigServerSomeipServiceInterfaceHzGetVINCfgParamId = 30255U; //HzGetVINCfgParam_method_hash


class HzAnswerAlive {
public:
    using Output = hozon::config::server::methods::HzAnswerAlive::Output;

    HzAnswerAlive(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
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

class HzDelParam {
public:
    using Output = hozon::config::server::methods::HzDelParam::Output;

    HzDelParam(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
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

class HzGetMonitorClients {
public:
    using Output = hozon::config::server::methods::HzGetMonitorClients::Output;

    HzGetMonitorClients(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
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

class HzGetParam {
public:
    using Output = hozon::config::server::methods::HzGetParam::Output;

    HzGetParam(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::String& clientName, const ::String& paramName, const ::UInt8& paramTypeIn)
    {
        return method_(clientName, paramName, paramTypeIn);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::String, ::UInt8> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::String, ::UInt8> method_;
};

class HzMonitorParam {
public:
    using Output = hozon::config::server::methods::HzMonitorParam::Output;

    HzMonitorParam(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
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

class HzSetParam {
public:
    using Output = hozon::config::server::methods::HzSetParam::Output;

    HzSetParam(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::String& clientName, const ::String& paramName, const ::String& paramValue, const ::UInt8& paramType, const ::Boolean& isPersist)
    {
        return method_(clientName, paramName, paramValue, paramType, isPersist);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::String, ::String, ::UInt8, ::Boolean> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::String, ::String, ::UInt8, ::Boolean> method_;
};

class HzUnMonitorParam {
public:
    using Output = hozon::config::server::methods::HzUnMonitorParam::Output;

    HzUnMonitorParam(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
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

class VehicleCfgUpdateResFromMcu {
public:
    using Output = hozon::config::server::methods::VehicleCfgUpdateResFromMcu::Output;

    VehicleCfgUpdateResFromMcu(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::UInt8& returnCode)
    {
        return method_(returnCode);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::UInt8> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::UInt8> method_;
};

class HzGetVehicleCfgParam {
public:
    using Output = hozon::config::server::methods::HzGetVehicleCfgParam::Output;

    HzGetVehicleCfgParam(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::String& ModuleName, const ::String& paramName)
    {
        return method_(ModuleName, paramName);
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

class HzGetVINCfgParam {
public:
    using Output = hozon::config::server::methods::HzGetVINCfgParam::Output;

    HzGetVINCfgParam(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::String& ModuleName, const ::String& paramName)
    {
        return method_(ModuleName, paramName);
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
} // namespace methods

class HzConfigServerSomeipServiceInterfaceProxy {
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

    virtual ~HzConfigServerSomeipServiceInterfaceProxy()
    {
        HzParamUpdateEvent.UnsetReceiveHandler();
        HzParamUpdateEvent.Unsubscribe();
        HzServerNotifyEvent.UnsetReceiveHandler();
        HzServerNotifyEvent.Unsubscribe();
        VehicleCfgUpdateToMcuEvent.UnsetReceiveHandler();
        VehicleCfgUpdateToMcuEvent.Unsubscribe();
    }

    explicit HzConfigServerSomeipServiceInterfaceProxy(const vrtf::vcc::api::types::HandleType &handle)
        : proxyAdapter(std::make_unique<ara::com::internal::proxy::ProxyAdapter>(::hozon::config::server::HzConfigServerSomeipServiceInterface::ServiceIdentifier, handle)),
          HzParamUpdateEvent(proxyAdapter->GetProxy(), events::HzConfigServerSomeipServiceInterfaceHzParamUpdateEventId, proxyAdapter->GetHandle(), ::hozon::config::server::HzConfigServerSomeipServiceInterface::ServiceIdentifier),
          HzServerNotifyEvent(proxyAdapter->GetProxy(), events::HzConfigServerSomeipServiceInterfaceHzServerNotifyEventId, proxyAdapter->GetHandle(), ::hozon::config::server::HzConfigServerSomeipServiceInterface::ServiceIdentifier),
          VehicleCfgUpdateToMcuEvent(proxyAdapter->GetProxy(), events::HzConfigServerSomeipServiceInterfaceVehicleCfgUpdateToMcuEventId, proxyAdapter->GetHandle(), ::hozon::config::server::HzConfigServerSomeipServiceInterface::ServiceIdentifier),
          HzAnswerAlive(proxyAdapter->GetProxy(), methods::HzConfigServerSomeipServiceInterfaceHzAnswerAliveId),
          HzDelParam(proxyAdapter->GetProxy(), methods::HzConfigServerSomeipServiceInterfaceHzDelParamId),
          HzGetMonitorClients(proxyAdapter->GetProxy(), methods::HzConfigServerSomeipServiceInterfaceHzGetMonitorClientsId),
          HzGetParam(proxyAdapter->GetProxy(), methods::HzConfigServerSomeipServiceInterfaceHzGetParamId),
          HzMonitorParam(proxyAdapter->GetProxy(), methods::HzConfigServerSomeipServiceInterfaceHzMonitorParamId),
          HzSetParam(proxyAdapter->GetProxy(), methods::HzConfigServerSomeipServiceInterfaceHzSetParamId),
          HzUnMonitorParam(proxyAdapter->GetProxy(), methods::HzConfigServerSomeipServiceInterfaceHzUnMonitorParamId),
          VehicleCfgUpdateResFromMcu(proxyAdapter->GetProxy(), methods::HzConfigServerSomeipServiceInterfaceVehicleCfgUpdateResFromMcuId),
          HzGetVehicleCfgParam(proxyAdapter->GetProxy(), methods::HzConfigServerSomeipServiceInterfaceHzGetVehicleCfgParamId),
          HzGetVINCfgParam(proxyAdapter->GetProxy(), methods::HzConfigServerSomeipServiceInterfaceHzGetVINCfgParamId){
            ara::core::Result<void> resultHzAnswerAlive = proxyAdapter->InitializeMethod<methods::HzAnswerAlive::Output>(methods::HzConfigServerSomeipServiceInterfaceHzAnswerAliveId);
            ThrowError(resultHzAnswerAlive);
            ara::core::Result<void> resultHzDelParam = proxyAdapter->InitializeMethod<methods::HzDelParam::Output>(methods::HzConfigServerSomeipServiceInterfaceHzDelParamId);
            ThrowError(resultHzDelParam);
            ara::core::Result<void> resultHzGetMonitorClients = proxyAdapter->InitializeMethod<methods::HzGetMonitorClients::Output>(methods::HzConfigServerSomeipServiceInterfaceHzGetMonitorClientsId);
            ThrowError(resultHzGetMonitorClients);
            ara::core::Result<void> resultHzGetParam = proxyAdapter->InitializeMethod<methods::HzGetParam::Output>(methods::HzConfigServerSomeipServiceInterfaceHzGetParamId);
            ThrowError(resultHzGetParam);
            ara::core::Result<void> resultHzMonitorParam = proxyAdapter->InitializeMethod<methods::HzMonitorParam::Output>(methods::HzConfigServerSomeipServiceInterfaceHzMonitorParamId);
            ThrowError(resultHzMonitorParam);
            ara::core::Result<void> resultHzSetParam = proxyAdapter->InitializeMethod<methods::HzSetParam::Output>(methods::HzConfigServerSomeipServiceInterfaceHzSetParamId);
            ThrowError(resultHzSetParam);
            ara::core::Result<void> resultHzUnMonitorParam = proxyAdapter->InitializeMethod<methods::HzUnMonitorParam::Output>(methods::HzConfigServerSomeipServiceInterfaceHzUnMonitorParamId);
            ThrowError(resultHzUnMonitorParam);
            ara::core::Result<void> resultVehicleCfgUpdateResFromMcu = proxyAdapter->InitializeMethod<methods::VehicleCfgUpdateResFromMcu::Output>(methods::HzConfigServerSomeipServiceInterfaceVehicleCfgUpdateResFromMcuId);
            ThrowError(resultVehicleCfgUpdateResFromMcu);
            ara::core::Result<void> resultHzGetVehicleCfgParam = proxyAdapter->InitializeMethod<methods::HzGetVehicleCfgParam::Output>(methods::HzConfigServerSomeipServiceInterfaceHzGetVehicleCfgParamId);
            ThrowError(resultHzGetVehicleCfgParam);
            ara::core::Result<void> resultHzGetVINCfgParam = proxyAdapter->InitializeMethod<methods::HzGetVINCfgParam::Output>(methods::HzConfigServerSomeipServiceInterfaceHzGetVINCfgParamId);
            ThrowError(resultHzGetVINCfgParam);
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

    HzConfigServerSomeipServiceInterfaceProxy(const HzConfigServerSomeipServiceInterfaceProxy&) = delete;
    HzConfigServerSomeipServiceInterfaceProxy& operator=(const HzConfigServerSomeipServiceInterfaceProxy&) = delete;

    HzConfigServerSomeipServiceInterfaceProxy(HzConfigServerSomeipServiceInterfaceProxy&&) = default;
    HzConfigServerSomeipServiceInterfaceProxy& operator=(HzConfigServerSomeipServiceInterfaceProxy&&) = default;
    HzConfigServerSomeipServiceInterfaceProxy(ConstructionToken&& token) noexcept
        : proxyAdapter(token.GetProxyAdapter()),
          HzParamUpdateEvent(proxyAdapter->GetProxy(), events::HzConfigServerSomeipServiceInterfaceHzParamUpdateEventId, proxyAdapter->GetHandle(), ::hozon::config::server::HzConfigServerSomeipServiceInterface::ServiceIdentifier),
          HzServerNotifyEvent(proxyAdapter->GetProxy(), events::HzConfigServerSomeipServiceInterfaceHzServerNotifyEventId, proxyAdapter->GetHandle(), ::hozon::config::server::HzConfigServerSomeipServiceInterface::ServiceIdentifier),
          VehicleCfgUpdateToMcuEvent(proxyAdapter->GetProxy(), events::HzConfigServerSomeipServiceInterfaceVehicleCfgUpdateToMcuEventId, proxyAdapter->GetHandle(), ::hozon::config::server::HzConfigServerSomeipServiceInterface::ServiceIdentifier),
          HzAnswerAlive(proxyAdapter->GetProxy(), methods::HzConfigServerSomeipServiceInterfaceHzAnswerAliveId),
          HzDelParam(proxyAdapter->GetProxy(), methods::HzConfigServerSomeipServiceInterfaceHzDelParamId),
          HzGetMonitorClients(proxyAdapter->GetProxy(), methods::HzConfigServerSomeipServiceInterfaceHzGetMonitorClientsId),
          HzGetParam(proxyAdapter->GetProxy(), methods::HzConfigServerSomeipServiceInterfaceHzGetParamId),
          HzMonitorParam(proxyAdapter->GetProxy(), methods::HzConfigServerSomeipServiceInterfaceHzMonitorParamId),
          HzSetParam(proxyAdapter->GetProxy(), methods::HzConfigServerSomeipServiceInterfaceHzSetParamId),
          HzUnMonitorParam(proxyAdapter->GetProxy(), methods::HzConfigServerSomeipServiceInterfaceHzUnMonitorParamId),
          VehicleCfgUpdateResFromMcu(proxyAdapter->GetProxy(), methods::HzConfigServerSomeipServiceInterfaceVehicleCfgUpdateResFromMcuId),
          HzGetVehicleCfgParam(proxyAdapter->GetProxy(), methods::HzConfigServerSomeipServiceInterfaceHzGetVehicleCfgParamId),
          HzGetVINCfgParam(proxyAdapter->GetProxy(), methods::HzConfigServerSomeipServiceInterfaceHzGetVINCfgParamId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        const vrtf::vcc::api::types::HandleType &handle)
    {
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> preProxyAdapter =
            std::make_unique<ara::com::internal::proxy::ProxyAdapter>(
               ::hozon::config::server::HzConfigServerSomeipServiceInterface::ServiceIdentifier, handle);
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            const methods::HzAnswerAlive HzAnswerAlive(preProxyAdapter->GetProxy(), methods::HzConfigServerSomeipServiceInterfaceHzAnswerAliveId);
            initResult = preProxyAdapter->InitializeMethod<methods::HzAnswerAlive::Output>(methods::HzConfigServerSomeipServiceInterfaceHzAnswerAliveId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::HzDelParam HzDelParam(preProxyAdapter->GetProxy(), methods::HzConfigServerSomeipServiceInterfaceHzDelParamId);
            initResult = preProxyAdapter->InitializeMethod<methods::HzDelParam::Output>(methods::HzConfigServerSomeipServiceInterfaceHzDelParamId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::HzGetMonitorClients HzGetMonitorClients(preProxyAdapter->GetProxy(), methods::HzConfigServerSomeipServiceInterfaceHzGetMonitorClientsId);
            initResult = preProxyAdapter->InitializeMethod<methods::HzGetMonitorClients::Output>(methods::HzConfigServerSomeipServiceInterfaceHzGetMonitorClientsId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::HzGetParam HzGetParam(preProxyAdapter->GetProxy(), methods::HzConfigServerSomeipServiceInterfaceHzGetParamId);
            initResult = preProxyAdapter->InitializeMethod<methods::HzGetParam::Output>(methods::HzConfigServerSomeipServiceInterfaceHzGetParamId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::HzMonitorParam HzMonitorParam(preProxyAdapter->GetProxy(), methods::HzConfigServerSomeipServiceInterfaceHzMonitorParamId);
            initResult = preProxyAdapter->InitializeMethod<methods::HzMonitorParam::Output>(methods::HzConfigServerSomeipServiceInterfaceHzMonitorParamId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::HzSetParam HzSetParam(preProxyAdapter->GetProxy(), methods::HzConfigServerSomeipServiceInterfaceHzSetParamId);
            initResult = preProxyAdapter->InitializeMethod<methods::HzSetParam::Output>(methods::HzConfigServerSomeipServiceInterfaceHzSetParamId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::HzUnMonitorParam HzUnMonitorParam(preProxyAdapter->GetProxy(), methods::HzConfigServerSomeipServiceInterfaceHzUnMonitorParamId);
            initResult = preProxyAdapter->InitializeMethod<methods::HzUnMonitorParam::Output>(methods::HzConfigServerSomeipServiceInterfaceHzUnMonitorParamId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::VehicleCfgUpdateResFromMcu VehicleCfgUpdateResFromMcu(preProxyAdapter->GetProxy(), methods::HzConfigServerSomeipServiceInterfaceVehicleCfgUpdateResFromMcuId);
            initResult = preProxyAdapter->InitializeMethod<methods::VehicleCfgUpdateResFromMcu::Output>(methods::HzConfigServerSomeipServiceInterfaceVehicleCfgUpdateResFromMcuId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::HzGetVehicleCfgParam HzGetVehicleCfgParam(preProxyAdapter->GetProxy(), methods::HzConfigServerSomeipServiceInterfaceHzGetVehicleCfgParamId);
            initResult = preProxyAdapter->InitializeMethod<methods::HzGetVehicleCfgParam::Output>(methods::HzConfigServerSomeipServiceInterfaceHzGetVehicleCfgParamId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::HzGetVINCfgParam HzGetVINCfgParam(preProxyAdapter->GetProxy(), methods::HzConfigServerSomeipServiceInterfaceHzGetVINCfgParamId);
            initResult = preProxyAdapter->InitializeMethod<methods::HzGetVINCfgParam::Output>(methods::HzConfigServerSomeipServiceInterfaceHzGetVINCfgParamId);
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
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::hozon::config::server::HzConfigServerSomeipServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType> handler,
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::hozon::config::server::HzConfigServerSomeipServiceInterface::ServiceIdentifier, specifier);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::hozon::config::server::HzConfigServerSomeipServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::hozon::config::server::HzConfigServerSomeipServiceInterface::ServiceIdentifier, specifier);
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
    events::HzParamUpdateEvent HzParamUpdateEvent;
    events::HzServerNotifyEvent HzServerNotifyEvent;
    events::VehicleCfgUpdateToMcuEvent VehicleCfgUpdateToMcuEvent;
    methods::HzAnswerAlive HzAnswerAlive;
    methods::HzDelParam HzDelParam;
    methods::HzGetMonitorClients HzGetMonitorClients;
    methods::HzGetParam HzGetParam;
    methods::HzMonitorParam HzMonitorParam;
    methods::HzSetParam HzSetParam;
    methods::HzUnMonitorParam HzUnMonitorParam;
    methods::VehicleCfgUpdateResFromMcu VehicleCfgUpdateResFromMcu;
    methods::HzGetVehicleCfgParam HzGetVehicleCfgParam;
    methods::HzGetVINCfgParam HzGetVINCfgParam;
};
} // namespace proxy
} // namespace server
} // namespace config
} // namespace hozon

#endif // HOZON_CONFIG_SERVER_HZCONFIGSERVERSOMEIPSERVICEINTERFACE_PROXY_H
