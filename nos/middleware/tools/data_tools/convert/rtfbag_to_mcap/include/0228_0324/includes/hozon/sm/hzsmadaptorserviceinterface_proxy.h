/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SM_HZSMADAPTORSERVICEINTERFACE_PROXY_H
#define HOZON_SM_HZSMADAPTORSERVICEINTERFACE_PROXY_H

#include "ara/com/internal/proxy/proxy_adapter.h"
#include "ara/com/internal/proxy/event_adapter.h"
#include "ara/com/internal/proxy/field_adapter.h"
#include "ara/com/internal/proxy/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/sm/hzsmadaptorserviceinterface_common.h"
#include <string>

namespace hozon {
namespace sm {
namespace proxy {
namespace events {
}

namespace fields {
    using MachineState = ara::com::internal::proxy::field::FieldAdapter<::String>;
    static constexpr ara::com::internal::EntityId HzSmAdaptorServiceInterfaceMachineStateId = 21022U; //MachineState_field_hash
}

namespace methods {
static constexpr ara::com::internal::EntityId HzSmAdaptorServiceInterfaceFuncGroupStateChangeId = 15833U; //FuncGroupStateChange_method_hash
static constexpr ara::com::internal::EntityId HzSmAdaptorServiceInterfaceMultiFuncGroupStateChangeId = 62434U; //MultiFuncGroupStateChange_method_hash
static constexpr ara::com::internal::EntityId HzSmAdaptorServiceInterfaceQueryFuncFroupStateId = 46362U; //QueryFuncFroupState_method_hash
static constexpr ara::com::internal::EntityId HzSmAdaptorServiceInterfaceRestartProcByNameId = 63703U; //RestartProcByName_method_hash


class FuncGroupStateChange {
public:
    using Output = hozon::sm::methods::FuncGroupStateChange::Output;

    FuncGroupStateChange(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::hozon::sm::FGStateChange& stateChange)
    {
        return method_(stateChange);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::hozon::sm::FGStateChange> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::hozon::sm::FGStateChange> method_;
};

class MultiFuncGroupStateChange {
public:
    using Output = hozon::sm::methods::MultiFuncGroupStateChange::Output;

    MultiFuncGroupStateChange(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::hozon::sm::FGStateChangeVector& stateChanges)
    {
        return method_(stateChanges);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::hozon::sm::FGStateChangeVector> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::hozon::sm::FGStateChangeVector> method_;
};

class QueryFuncFroupState {
public:
    using Output = hozon::sm::methods::QueryFuncFroupState::Output;

    QueryFuncFroupState(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::String& fgName)
    {
        return method_(fgName);
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

class RestartProcByName {
public:
    using Output = void;

    RestartProcByName(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    void operator()(const ::String& procName)
    {
        method_(procName);
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

class HzSmAdaptorServiceInterfaceProxy {
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

    virtual ~HzSmAdaptorServiceInterfaceProxy()
    {
    }

    explicit HzSmAdaptorServiceInterfaceProxy(const vrtf::vcc::api::types::HandleType &handle)
        : proxyAdapter(std::make_unique<ara::com::internal::proxy::ProxyAdapter>(::hozon::sm::HzSmAdaptorServiceInterface::ServiceIdentifier, handle)),
          FuncGroupStateChange(proxyAdapter->GetProxy(), methods::HzSmAdaptorServiceInterfaceFuncGroupStateChangeId),
          MultiFuncGroupStateChange(proxyAdapter->GetProxy(), methods::HzSmAdaptorServiceInterfaceMultiFuncGroupStateChangeId),
          QueryFuncFroupState(proxyAdapter->GetProxy(), methods::HzSmAdaptorServiceInterfaceQueryFuncFroupStateId),
          RestartProcByName(proxyAdapter->GetProxy(), methods::HzSmAdaptorServiceInterfaceRestartProcByNameId),
          MachineState(proxyAdapter->GetProxy(), fields::HzSmAdaptorServiceInterfaceMachineStateId, proxyAdapter->GetHandle(), ::hozon::sm::HzSmAdaptorServiceInterface::ServiceIdentifier) {
            ara::core::Result<void> resultFuncGroupStateChange = proxyAdapter->InitializeMethod<methods::FuncGroupStateChange::Output>(methods::HzSmAdaptorServiceInterfaceFuncGroupStateChangeId);
            ThrowError(resultFuncGroupStateChange);
            ara::core::Result<void> resultMultiFuncGroupStateChange = proxyAdapter->InitializeMethod<methods::MultiFuncGroupStateChange::Output>(methods::HzSmAdaptorServiceInterfaceMultiFuncGroupStateChangeId);
            ThrowError(resultMultiFuncGroupStateChange);
            ara::core::Result<void> resultQueryFuncFroupState = proxyAdapter->InitializeMethod<methods::QueryFuncFroupState::Output>(methods::HzSmAdaptorServiceInterfaceQueryFuncFroupStateId);
            ThrowError(resultQueryFuncFroupState);
            ara::core::Result<void> resultRestartProcByName = proxyAdapter->InitializeMethod<methods::RestartProcByName::Output>(methods::HzSmAdaptorServiceInterfaceRestartProcByNameId);
            ThrowError(resultRestartProcByName);
            ara::core::Result<void> resultMachineState = proxyAdapter->InitializeField<::String>(MachineState);
            ThrowError(resultMachineState);
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

    HzSmAdaptorServiceInterfaceProxy(const HzSmAdaptorServiceInterfaceProxy&) = delete;
    HzSmAdaptorServiceInterfaceProxy& operator=(const HzSmAdaptorServiceInterfaceProxy&) = delete;

    HzSmAdaptorServiceInterfaceProxy(HzSmAdaptorServiceInterfaceProxy&&) = default;
    HzSmAdaptorServiceInterfaceProxy& operator=(HzSmAdaptorServiceInterfaceProxy&&) = default;
    HzSmAdaptorServiceInterfaceProxy(ConstructionToken&& token) noexcept
        : proxyAdapter(token.GetProxyAdapter()),
          FuncGroupStateChange(proxyAdapter->GetProxy(), methods::HzSmAdaptorServiceInterfaceFuncGroupStateChangeId),
          MultiFuncGroupStateChange(proxyAdapter->GetProxy(), methods::HzSmAdaptorServiceInterfaceMultiFuncGroupStateChangeId),
          QueryFuncFroupState(proxyAdapter->GetProxy(), methods::HzSmAdaptorServiceInterfaceQueryFuncFroupStateId),
          RestartProcByName(proxyAdapter->GetProxy(), methods::HzSmAdaptorServiceInterfaceRestartProcByNameId),
          MachineState(proxyAdapter->GetProxy(), fields::HzSmAdaptorServiceInterfaceMachineStateId, proxyAdapter->GetHandle(), ::hozon::sm::HzSmAdaptorServiceInterface::ServiceIdentifier) {
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        const vrtf::vcc::api::types::HandleType &handle)
    {
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> preProxyAdapter =
            std::make_unique<ara::com::internal::proxy::ProxyAdapter>(
               ::hozon::sm::HzSmAdaptorServiceInterface::ServiceIdentifier, handle);
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            const methods::FuncGroupStateChange FuncGroupStateChange(preProxyAdapter->GetProxy(), methods::HzSmAdaptorServiceInterfaceFuncGroupStateChangeId);
            initResult = preProxyAdapter->InitializeMethod<methods::FuncGroupStateChange::Output>(methods::HzSmAdaptorServiceInterfaceFuncGroupStateChangeId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::MultiFuncGroupStateChange MultiFuncGroupStateChange(preProxyAdapter->GetProxy(), methods::HzSmAdaptorServiceInterfaceMultiFuncGroupStateChangeId);
            initResult = preProxyAdapter->InitializeMethod<methods::MultiFuncGroupStateChange::Output>(methods::HzSmAdaptorServiceInterfaceMultiFuncGroupStateChangeId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::QueryFuncFroupState QueryFuncFroupState(preProxyAdapter->GetProxy(), methods::HzSmAdaptorServiceInterfaceQueryFuncFroupStateId);
            initResult = preProxyAdapter->InitializeMethod<methods::QueryFuncFroupState::Output>(methods::HzSmAdaptorServiceInterfaceQueryFuncFroupStateId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::RestartProcByName RestartProcByName(preProxyAdapter->GetProxy(), methods::HzSmAdaptorServiceInterfaceRestartProcByNameId);
            initResult = preProxyAdapter->InitializeMethod<methods::RestartProcByName::Output>(methods::HzSmAdaptorServiceInterfaceRestartProcByNameId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            fields::MachineState MachineState(preProxyAdapter->GetProxy(), fields::HzSmAdaptorServiceInterfaceMachineStateId, handle, ::hozon::sm::HzSmAdaptorServiceInterface::ServiceIdentifier);
            initResult = preProxyAdapter->InitializeField<::String>(MachineState);
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
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::hozon::sm::HzSmAdaptorServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType> handler,
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::hozon::sm::HzSmAdaptorServiceInterface::ServiceIdentifier, specifier);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::hozon::sm::HzSmAdaptorServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::hozon::sm::HzSmAdaptorServiceInterface::ServiceIdentifier, specifier);
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
    methods::FuncGroupStateChange FuncGroupStateChange;
    methods::MultiFuncGroupStateChange MultiFuncGroupStateChange;
    methods::QueryFuncFroupState QueryFuncFroupState;
    methods::RestartProcByName RestartProcByName;
    fields::MachineState MachineState;
};
} // namespace proxy
} // namespace sm
} // namespace hozon

#endif // HOZON_SM_HZSMADAPTORSERVICEINTERFACE_PROXY_H
