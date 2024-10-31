/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_SM_TRIGGERINOUT_PROXY_H
#define ARA_SM_TRIGGERINOUT_PROXY_H

#include "ara/com/internal/proxy/proxy_adapter.h"
#include "ara/com/internal/proxy/event_adapter.h"
#include "ara/com/internal/proxy/field_adapter.h"
#include "ara/com/internal/proxy/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "ara/sm/triggerinout_common.h"
#include <string>

namespace ara {
namespace sm {
namespace proxy {
namespace events {
}

namespace fields {
    using Notifier = ara::com::internal::proxy::field::FieldAdapter<::ara::sm::TriggerDataType>;
    using Trigger = ara::com::internal::proxy::field::FieldAdapter<::ara::sm::TriggerDataType>;
    static constexpr ara::com::internal::EntityId TriggerInOutNotifierId = 54564U; //Notifier_field_hash
    static constexpr ara::com::internal::EntityId TriggerInOutNotifierGetterId = 21625U; //Notifier_getter_hash
    static constexpr ara::com::internal::EntityId TriggerInOutTriggerId = 24507U; //Trigger_field_hash
    static constexpr ara::com::internal::EntityId TriggerInOutTriggerSetterId = 57439U; //Trigger_setter_hash
}

namespace methods {
static constexpr ara::com::internal::EntityId TriggerInOutAcquireFunctionGroupInfoId = 36548U; //AcquireFunctionGroupInfo_method_hash
static constexpr ara::com::internal::EntityId TriggerInOutProcessSyncRequestId = 8148U; //ProcessSyncRequest_method_hash
static constexpr ara::com::internal::EntityId TriggerInOutResetSystemId = 43154U; //ResetSystem_method_hash
static constexpr ara::com::internal::EntityId TriggerInOutProcessAsyncRequestId = 477U; //ProcessAsyncRequest_method_hash
static constexpr ara::com::internal::EntityId TriggerInOutShutdownSystemId = 3126U; //ShutdownSystem_method_hash


class AcquireFunctionGroupInfo {
public:
    using Output = ara::sm::methods::AcquireFunctionGroupInfo::Output;

    AcquireFunctionGroupInfo(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
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

class ProcessSyncRequest {
public:
    using Output = ara::sm::methods::ProcessSyncRequest::Output;

    ProcessSyncRequest(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::ara::sm::StateTransitionVec& stateTrans)
    {
        return method_(stateTrans);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::ara::sm::StateTransitionVec> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::ara::sm::StateTransitionVec> method_;
};

class ResetSystem {
public:
    using Output = ara::sm::methods::ResetSystem::Output;

    ResetSystem(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::ara::sm::ResetCode& resetParams, const ::String& user, const ::ara::sm::ResetCause& resetReason)
    {
        return method_(resetParams, user, resetReason);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::ara::sm::ResetCode, ::String, ::ara::sm::ResetCause> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::ara::sm::ResetCode, ::String, ::ara::sm::ResetCause> method_;
};

class ProcessAsyncRequest {
public:
    using Output = void;

    ProcessAsyncRequest(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    void operator()(const ::ara::sm::StateTransitionVec& stateTrans)
    {
        method_(stateTrans);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::ara::sm::StateTransitionVec> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::ara::sm::StateTransitionVec> method_;
};

class ShutdownSystem {
public:
    using Output = ara::sm::methods::ShutdownSystem::Output;

    ShutdownSystem(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::uint8_t& shutdownParams)
    {
        return method_(shutdownParams);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::uint8_t> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::uint8_t> method_;
};
} // namespace methods

class TriggerInOutProxy {
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

    virtual ~TriggerInOutProxy()
    {
    }

    explicit TriggerInOutProxy(const vrtf::vcc::api::types::HandleType &handle)
        : proxyAdapter(std::make_unique<ara::com::internal::proxy::ProxyAdapter>(::ara::sm::TriggerInOut::ServiceIdentifier, handle)),
          AcquireFunctionGroupInfo(proxyAdapter->GetProxy(), methods::TriggerInOutAcquireFunctionGroupInfoId),
          ProcessSyncRequest(proxyAdapter->GetProxy(), methods::TriggerInOutProcessSyncRequestId),
          ResetSystem(proxyAdapter->GetProxy(), methods::TriggerInOutResetSystemId),
          ProcessAsyncRequest(proxyAdapter->GetProxy(), methods::TriggerInOutProcessAsyncRequestId),
          ShutdownSystem(proxyAdapter->GetProxy(), methods::TriggerInOutShutdownSystemId),
          Notifier(proxyAdapter->GetProxy(), fields::TriggerInOutNotifierId, proxyAdapter->GetHandle(), ::ara::sm::TriggerInOut::ServiceIdentifier),
          Trigger(proxyAdapter->GetProxy(), fields::TriggerInOutTriggerId, proxyAdapter->GetHandle(), ::ara::sm::TriggerInOut::ServiceIdentifier) {
            ara::core::Result<void> resultAcquireFunctionGroupInfo = proxyAdapter->InitializeMethod<methods::AcquireFunctionGroupInfo::Output>(methods::TriggerInOutAcquireFunctionGroupInfoId);
            ThrowError(resultAcquireFunctionGroupInfo);
            ara::core::Result<void> resultProcessSyncRequest = proxyAdapter->InitializeMethod<methods::ProcessSyncRequest::Output>(methods::TriggerInOutProcessSyncRequestId);
            ThrowError(resultProcessSyncRequest);
            ara::core::Result<void> resultResetSystem = proxyAdapter->InitializeMethod<methods::ResetSystem::Output>(methods::TriggerInOutResetSystemId);
            ThrowError(resultResetSystem);
            ara::core::Result<void> resultProcessAsyncRequest = proxyAdapter->InitializeMethod<methods::ProcessAsyncRequest::Output>(methods::TriggerInOutProcessAsyncRequestId);
            ThrowError(resultProcessAsyncRequest);
            ara::core::Result<void> resultShutdownSystem = proxyAdapter->InitializeMethod<methods::ShutdownSystem::Output>(methods::TriggerInOutShutdownSystemId);
            ThrowError(resultShutdownSystem);
            Notifier.SetGetterEntityId(fields::TriggerInOutNotifierGetterId);
            ara::core::Result<void> resultNotifier = proxyAdapter->InitializeField<::ara::sm::TriggerDataType>(Notifier);
            ThrowError(resultNotifier);
            Trigger.SetSetterEntityId(fields::TriggerInOutTriggerSetterId);
            ara::core::Result<void> resultTrigger = proxyAdapter->InitializeField<::ara::sm::TriggerDataType>(Trigger);
            ThrowError(resultTrigger);
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

    TriggerInOutProxy(const TriggerInOutProxy&) = delete;
    TriggerInOutProxy& operator=(const TriggerInOutProxy&) = delete;

    TriggerInOutProxy(TriggerInOutProxy&&) = default;
    TriggerInOutProxy& operator=(TriggerInOutProxy&&) = default;
    TriggerInOutProxy(ConstructionToken&& token) noexcept
        : proxyAdapter(token.GetProxyAdapter()),
          AcquireFunctionGroupInfo(proxyAdapter->GetProxy(), methods::TriggerInOutAcquireFunctionGroupInfoId),
          ProcessSyncRequest(proxyAdapter->GetProxy(), methods::TriggerInOutProcessSyncRequestId),
          ResetSystem(proxyAdapter->GetProxy(), methods::TriggerInOutResetSystemId),
          ProcessAsyncRequest(proxyAdapter->GetProxy(), methods::TriggerInOutProcessAsyncRequestId),
          ShutdownSystem(proxyAdapter->GetProxy(), methods::TriggerInOutShutdownSystemId),
          Notifier(proxyAdapter->GetProxy(), fields::TriggerInOutNotifierId, proxyAdapter->GetHandle(), ::ara::sm::TriggerInOut::ServiceIdentifier),
          Trigger(proxyAdapter->GetProxy(), fields::TriggerInOutTriggerId, proxyAdapter->GetHandle(), ::ara::sm::TriggerInOut::ServiceIdentifier) {
        Notifier.SetGetterEntityId(fields::TriggerInOutNotifierGetterId);
        Trigger.SetSetterEntityId(fields::TriggerInOutTriggerSetterId);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        const vrtf::vcc::api::types::HandleType &handle)
    {
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> preProxyAdapter =
            std::make_unique<ara::com::internal::proxy::ProxyAdapter>(
               ::ara::sm::TriggerInOut::ServiceIdentifier, handle);
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            const methods::AcquireFunctionGroupInfo AcquireFunctionGroupInfo(preProxyAdapter->GetProxy(), methods::TriggerInOutAcquireFunctionGroupInfoId);
            initResult = preProxyAdapter->InitializeMethod<methods::AcquireFunctionGroupInfo::Output>(methods::TriggerInOutAcquireFunctionGroupInfoId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::ProcessSyncRequest ProcessSyncRequest(preProxyAdapter->GetProxy(), methods::TriggerInOutProcessSyncRequestId);
            initResult = preProxyAdapter->InitializeMethod<methods::ProcessSyncRequest::Output>(methods::TriggerInOutProcessSyncRequestId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::ResetSystem ResetSystem(preProxyAdapter->GetProxy(), methods::TriggerInOutResetSystemId);
            initResult = preProxyAdapter->InitializeMethod<methods::ResetSystem::Output>(methods::TriggerInOutResetSystemId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::ProcessAsyncRequest ProcessAsyncRequest(preProxyAdapter->GetProxy(), methods::TriggerInOutProcessAsyncRequestId);
            initResult = preProxyAdapter->InitializeMethod<methods::ProcessAsyncRequest::Output>(methods::TriggerInOutProcessAsyncRequestId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::ShutdownSystem ShutdownSystem(preProxyAdapter->GetProxy(), methods::TriggerInOutShutdownSystemId);
            initResult = preProxyAdapter->InitializeMethod<methods::ShutdownSystem::Output>(methods::TriggerInOutShutdownSystemId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            fields::Notifier Notifier(preProxyAdapter->GetProxy(), fields::TriggerInOutNotifierId, handle, ::ara::sm::TriggerInOut::ServiceIdentifier);
            Notifier.SetGetterEntityId(fields::TriggerInOutNotifierGetterId);
            initResult = preProxyAdapter->InitializeField<::ara::sm::TriggerDataType>(Notifier);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            fields::Trigger Trigger(preProxyAdapter->GetProxy(), fields::TriggerInOutTriggerId, handle, ::ara::sm::TriggerInOut::ServiceIdentifier);
            Trigger.SetSetterEntityId(fields::TriggerInOutTriggerSetterId);
            initResult = preProxyAdapter->InitializeField<::ara::sm::TriggerDataType>(Trigger);
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
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::ara::sm::TriggerInOut::ServiceIdentifier, instance);
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType> handler,
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::ara::sm::TriggerInOut::ServiceIdentifier, specifier);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::ara::sm::TriggerInOut::ServiceIdentifier, instance);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::ara::sm::TriggerInOut::ServiceIdentifier, specifier);
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
    methods::AcquireFunctionGroupInfo AcquireFunctionGroupInfo;
    methods::ProcessSyncRequest ProcessSyncRequest;
    methods::ResetSystem ResetSystem;
    methods::ProcessAsyncRequest ProcessAsyncRequest;
    methods::ShutdownSystem ShutdownSystem;
    fields::Notifier Notifier;
    fields::Trigger Trigger;
};
} // namespace proxy
} // namespace sm
} // namespace ara

#endif // ARA_SM_TRIGGERINOUT_PROXY_H
