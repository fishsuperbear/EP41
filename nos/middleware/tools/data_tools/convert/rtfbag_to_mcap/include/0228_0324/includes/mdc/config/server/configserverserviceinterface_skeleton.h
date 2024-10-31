/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_CONFIG_SERVER_CONFIGSERVERSERVICEINTERFACE_SKELETON_H
#define MDC_CONFIG_SERVER_CONFIGSERVERSERVICEINTERFACE_SKELETON_H

#include "ara/com/internal/skeleton/skeleton_adapter.h"
#include "ara/com/internal/skeleton/event_adapter.h"
#include "ara/com/internal/skeleton/field_adapter.h"
#include "ara/com/internal/skeleton/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "mdc/config/server/configserverserviceinterface_common.h"
#include <cstdint>

namespace mdc {
namespace config {
namespace server {
namespace skeleton {
namespace events
{
    using ParamUpdateEvent = ara::com::internal::skeleton::event::EventAdapter<::mdc::config::server::ParamUpdateData>;
    using ServerNotifyEvent = ara::com::internal::skeleton::event::EventAdapter<::mdc::config::server::ServerNotifyData>;
    static constexpr ara::com::internal::EntityId ConfigServerServiceInterfaceParamUpdateEventId = 35207U; //ParamUpdateEvent_event_hash
    static constexpr ara::com::internal::EntityId ConfigServerServiceInterfaceServerNotifyEventId = 27155U; //ServerNotifyEvent_event_hash
}

namespace methods
{
    using AnswerAliveHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using DelParamHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using GetMonitorClientsHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using GetParamHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using MonitorParamHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using SetParamHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using UnMonitorParamHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using InitClientHandle = ara::com::internal::skeleton::method::MethodAdapter;
    static constexpr ara::com::internal::EntityId ConfigServerServiceInterfaceAnswerAliveId = 30329U; //AnswerAlive_method_hash
    static constexpr ara::com::internal::EntityId ConfigServerServiceInterfaceDelParamId = 56227U; //DelParam_method_hash
    static constexpr ara::com::internal::EntityId ConfigServerServiceInterfaceGetMonitorClientsId = 42802U; //GetMonitorClients_method_hash
    static constexpr ara::com::internal::EntityId ConfigServerServiceInterfaceGetParamId = 5505U; //GetParam_method_hash
    static constexpr ara::com::internal::EntityId ConfigServerServiceInterfaceMonitorParamId = 3875U; //MonitorParam_method_hash
    static constexpr ara::com::internal::EntityId ConfigServerServiceInterfaceSetParamId = 31959U; //SetParam_method_hash
    static constexpr ara::com::internal::EntityId ConfigServerServiceInterfaceUnMonitorParamId = 36659U; //UnMonitorParam_method_hash
    static constexpr ara::com::internal::EntityId ConfigServerServiceInterfaceInitClientId = 55067U; //InitClient_method_hash
}

namespace fields
{
}

class ConfigServerServiceInterfaceSkeleton {
private:
    std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> skeletonAdapter;
    void ConstructSkeleton(const ara::com::MethodCallProcessingMode mode)
    {
        if (mode == ara::com::MethodCallProcessingMode::kEvent) {
            if (!(skeletonAdapter->SetMethodThreadNumber(skeletonAdapter->GetMethodThreadNumber(8U), 1024U))) {
#ifndef NOT_SUPPORT_EXCEPTIONS
                ara::core::ErrorCode errorcode(ara::com::ComErrc::kNetworkBindingFailure);
                throw ara::com::ComException(std::move(errorcode));
#else
                std::cerr << "Error: Not support exception, create skeleton failed!"<< std::endl;
#endif
            }
        }
        const ara::core::Result<void> resultParamUpdateEvent = skeletonAdapter->InitializeEvent(ParamUpdateEvent);
        ThrowError(resultParamUpdateEvent);
        const ara::core::Result<void> resultServerNotifyEvent = skeletonAdapter->InitializeEvent(ServerNotifyEvent);
        ThrowError(resultServerNotifyEvent);
        const ara::core::Result<void> resultAnswerAlive = skeletonAdapter->InitializeMethod<ara::core::Future<AnswerAliveOutput>>(methods::ConfigServerServiceInterfaceAnswerAliveId);
        ThrowError(resultAnswerAlive);
        const ara::core::Result<void> resultDelParam = skeletonAdapter->InitializeMethod<ara::core::Future<DelParamOutput>>(methods::ConfigServerServiceInterfaceDelParamId);
        ThrowError(resultDelParam);
        const ara::core::Result<void> resultGetMonitorClients = skeletonAdapter->InitializeMethod<ara::core::Future<GetMonitorClientsOutput>>(methods::ConfigServerServiceInterfaceGetMonitorClientsId);
        ThrowError(resultGetMonitorClients);
        const ara::core::Result<void> resultGetParam = skeletonAdapter->InitializeMethod<ara::core::Future<GetParamOutput>>(methods::ConfigServerServiceInterfaceGetParamId);
        ThrowError(resultGetParam);
        const ara::core::Result<void> resultMonitorParam = skeletonAdapter->InitializeMethod<ara::core::Future<MonitorParamOutput>>(methods::ConfigServerServiceInterfaceMonitorParamId);
        ThrowError(resultMonitorParam);
        const ara::core::Result<void> resultSetParam = skeletonAdapter->InitializeMethod<ara::core::Future<SetParamOutput>>(methods::ConfigServerServiceInterfaceSetParamId);
        ThrowError(resultSetParam);
        const ara::core::Result<void> resultUnMonitorParam = skeletonAdapter->InitializeMethod<ara::core::Future<UnMonitorParamOutput>>(methods::ConfigServerServiceInterfaceUnMonitorParamId);
        ThrowError(resultUnMonitorParam);
        const ara::core::Result<void> resultInitClient = skeletonAdapter->InitializeMethod<ara::core::Future<InitClientOutput>>(methods::ConfigServerServiceInterfaceInitClientId);
        ThrowError(resultInitClient);
    }

    ConfigServerServiceInterfaceSkeleton& operator=(const ConfigServerServiceInterfaceSkeleton&) = delete;

    static void ThrowError(const ara::core::Result<void>& result)
    {
        if (!(result.HasValue())) {
#ifndef NOT_SUPPORT_EXCEPTIONS
            ara::core::ErrorCode errorcode(result.Error());
            throw ara::com::ComException(std::move(errorcode));
#else
            std::cerr << "Error: Not support exception, create skeleton failed!"<< std::endl;
#endif
        }
    }
public:
    using AnswerAliveOutput = mdc::config::server::methods::AnswerAlive::Output;
    
    using DelParamOutput = mdc::config::server::methods::DelParam::Output;
    
    using GetMonitorClientsOutput = mdc::config::server::methods::GetMonitorClients::Output;
    
    using GetParamOutput = mdc::config::server::methods::GetParam::Output;
    
    using MonitorParamOutput = mdc::config::server::methods::MonitorParam::Output;
    
    using SetParamOutput = mdc::config::server::methods::SetParam::Output;
    
    using UnMonitorParamOutput = mdc::config::server::methods::UnMonitorParam::Output;
    
    using InitClientOutput = mdc::config::server::methods::InitClient::Output;
    
    class ConstructionToken {
    public:
        explicit ConstructionToken(std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter>& skeleton)
            : ptr(std::move(skeleton)) {}
        explicit ConstructionToken(std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter>&& skeleton)
            : ptr(std::move(skeleton)) {}
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
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> GetSkeletonAdapter()
        {
            return std::move(ptr);
        }
    private:
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> ptr;
    };
    explicit ConfigServerServiceInterfaceSkeleton(const ara::com::InstanceIdentifier& instanceId,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        : skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::mdc::config::server::ConfigServerServiceInterface::ServiceIdentifier, instanceId, mode)),
          ParamUpdateEvent(skeletonAdapter->GetSkeleton(), events::ConfigServerServiceInterfaceParamUpdateEventId),
          ServerNotifyEvent(skeletonAdapter->GetSkeleton(), events::ConfigServerServiceInterfaceServerNotifyEventId),
          AnswerAliveHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceAnswerAliveId),
          DelParamHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceDelParamId),
          GetMonitorClientsHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceGetMonitorClientsId),
          GetParamHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceGetParamId),
          MonitorParamHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceMonitorParamId),
          SetParamHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceSetParamId),
          UnMonitorParamHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceUnMonitorParamId),
          InitClientHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceInitClientId){
        ConstructSkeleton(mode);
    }

    explicit ConfigServerServiceInterfaceSkeleton(const ara::core::InstanceSpecifier& instanceSpec,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::mdc::config::server::ConfigServerServiceInterface::ServiceIdentifier, instanceSpec, mode)),
          ParamUpdateEvent(skeletonAdapter->GetSkeleton(), events::ConfigServerServiceInterfaceParamUpdateEventId),
          ServerNotifyEvent(skeletonAdapter->GetSkeleton(), events::ConfigServerServiceInterfaceServerNotifyEventId),
          AnswerAliveHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceAnswerAliveId),
          DelParamHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceDelParamId),
          GetMonitorClientsHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceGetMonitorClientsId),
          GetParamHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceGetParamId),
          MonitorParamHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceMonitorParamId),
          SetParamHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceSetParamId),
          UnMonitorParamHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceUnMonitorParamId),
          InitClientHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceInitClientId){
        ConstructSkeleton(mode);
    }

    explicit ConfigServerServiceInterfaceSkeleton(const ara::com::InstanceIdentifierContainer instanceContainer,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::mdc::config::server::ConfigServerServiceInterface::ServiceIdentifier, instanceContainer, mode)),
          ParamUpdateEvent(skeletonAdapter->GetSkeleton(), events::ConfigServerServiceInterfaceParamUpdateEventId),
          ServerNotifyEvent(skeletonAdapter->GetSkeleton(), events::ConfigServerServiceInterfaceServerNotifyEventId),
          AnswerAliveHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceAnswerAliveId),
          DelParamHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceDelParamId),
          GetMonitorClientsHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceGetMonitorClientsId),
          GetParamHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceGetParamId),
          MonitorParamHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceMonitorParamId),
          SetParamHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceSetParamId),
          UnMonitorParamHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceUnMonitorParamId),
          InitClientHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceInitClientId){
        ConstructSkeleton(mode);
    }

    ConfigServerServiceInterfaceSkeleton(const ConfigServerServiceInterfaceSkeleton&) = delete;

    ConfigServerServiceInterfaceSkeleton(ConfigServerServiceInterfaceSkeleton&&) = default;
    ConfigServerServiceInterfaceSkeleton& operator=(ConfigServerServiceInterfaceSkeleton&&) = default;
    ConfigServerServiceInterfaceSkeleton(ConstructionToken&& token) noexcept 
        : skeletonAdapter(token.GetSkeletonAdapter()),
          ParamUpdateEvent(skeletonAdapter->GetSkeleton(), events::ConfigServerServiceInterfaceParamUpdateEventId),
          ServerNotifyEvent(skeletonAdapter->GetSkeleton(), events::ConfigServerServiceInterfaceServerNotifyEventId),
          AnswerAliveHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceAnswerAliveId),
          DelParamHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceDelParamId),
          GetMonitorClientsHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceGetMonitorClientsId),
          GetParamHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceGetParamId),
          MonitorParamHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceMonitorParamId),
          SetParamHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceSetParamId),
          UnMonitorParamHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceUnMonitorParamId),
          InitClientHandle(skeletonAdapter->GetSkeleton(), methods::ConfigServerServiceInterfaceInitClientId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifier instanceId,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::mdc::config::server::ConfigServerServiceInterface::ServiceIdentifier, instanceId, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::core::InstanceSpecifier instanceSpec,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::mdc::config::server::ConfigServerServiceInterface::ServiceIdentifier, instanceSpec, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifierContainer instanceIdContainer,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::mdc::config::server::ConfigServerServiceInterface::ServiceIdentifier, instanceIdContainer, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> PreConstructResult(
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter>& preSkeletonAdapter, const ara::com::MethodCallProcessingMode mode)
    {
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            if (mode == ara::com::MethodCallProcessingMode::kEvent) {
                if(!preSkeletonAdapter->SetMethodThreadNumber(preSkeletonAdapter->GetMethodThreadNumber(8U), 1024U)) {
                    result = false;
                    ara::core::ErrorCode errorcode(ara::com::ComErrc::kNetworkBindingFailure);
                    initResult.EmplaceError(errorcode);
                    break;
                }
            }
            const events::ParamUpdateEvent ParamUpdateEvent(preSkeletonAdapter->GetSkeleton(), events::ConfigServerServiceInterfaceParamUpdateEventId);
            initResult = preSkeletonAdapter->InitializeEvent(ParamUpdateEvent);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const events::ServerNotifyEvent ServerNotifyEvent(preSkeletonAdapter->GetSkeleton(), events::ConfigServerServiceInterfaceServerNotifyEventId);
            initResult = preSkeletonAdapter->InitializeEvent(ServerNotifyEvent);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<AnswerAliveOutput>>(methods::ConfigServerServiceInterfaceAnswerAliveId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<DelParamOutput>>(methods::ConfigServerServiceInterfaceDelParamId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<GetMonitorClientsOutput>>(methods::ConfigServerServiceInterfaceGetMonitorClientsId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<GetParamOutput>>(methods::ConfigServerServiceInterfaceGetParamId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<MonitorParamOutput>>(methods::ConfigServerServiceInterfaceMonitorParamId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<SetParamOutput>>(methods::ConfigServerServiceInterfaceSetParamId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<UnMonitorParamOutput>>(methods::ConfigServerServiceInterfaceUnMonitorParamId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<InitClientOutput>>(methods::ConfigServerServiceInterfaceInitClientId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
        } while(false);
        
        if (result) {
            ConstructionToken token(std::move(preSkeletonAdapter));
            return ara::core::Result<ConstructionToken>(std::move(token));
        } else {
            ConstructionToken token(std::move(preSkeletonAdapter));
            ara::core::Result<ConstructionToken> preResult(std::move(token));
            const ara::core::ErrorCode errorcode(initResult.Error());
            preResult.EmplaceError(errorcode);
            return preResult;
        }
    }

    virtual ~ConfigServerServiceInterfaceSkeleton()
    {
        StopOfferService();
    }

    void OfferService()
    {
        skeletonAdapter->RegisterE2EErrorHandler(&ConfigServerServiceInterfaceSkeleton::E2EErrorHandler, *this);
        skeletonAdapter->RegisterMethod(&ConfigServerServiceInterfaceSkeleton::AnswerAlive, *this, methods::ConfigServerServiceInterfaceAnswerAliveId);
        skeletonAdapter->RegisterMethod(&ConfigServerServiceInterfaceSkeleton::DelParam, *this, methods::ConfigServerServiceInterfaceDelParamId);
        skeletonAdapter->RegisterMethod(&ConfigServerServiceInterfaceSkeleton::GetMonitorClients, *this, methods::ConfigServerServiceInterfaceGetMonitorClientsId);
        skeletonAdapter->RegisterMethod(&ConfigServerServiceInterfaceSkeleton::GetParam, *this, methods::ConfigServerServiceInterfaceGetParamId);
        skeletonAdapter->RegisterMethod(&ConfigServerServiceInterfaceSkeleton::MonitorParam, *this, methods::ConfigServerServiceInterfaceMonitorParamId);
        skeletonAdapter->RegisterMethod(&ConfigServerServiceInterfaceSkeleton::SetParam, *this, methods::ConfigServerServiceInterfaceSetParamId);
        skeletonAdapter->RegisterMethod(&ConfigServerServiceInterfaceSkeleton::UnMonitorParam, *this, methods::ConfigServerServiceInterfaceUnMonitorParamId);
        skeletonAdapter->RegisterMethod(&ConfigServerServiceInterfaceSkeleton::InitClient, *this, methods::ConfigServerServiceInterfaceInitClientId);
        skeletonAdapter->OfferService();
    }
    void StopOfferService()
    {
        skeletonAdapter->StopOfferService();
    }
    ara::core::Future<bool> ProcessNextMethodCall()
    {
        return skeletonAdapter->ProcessNextMethodCall();
    }
    bool SetMethodThreadNumber(const std::uint16_t& number, const std::uint16_t& queueSize)
    {
        return skeletonAdapter->SetMethodThreadNumber(number, queueSize);
    }
    virtual ara::core::Future<AnswerAliveOutput> AnswerAlive(const ::String& clientName) = 0;
    virtual ara::core::Future<DelParamOutput> DelParam(const ::String& clientName, const ::String& paramName) = 0;
    virtual ara::core::Future<GetMonitorClientsOutput> GetMonitorClients(const ::String& clientName, const ::String& paramName) = 0;
    virtual ara::core::Future<GetParamOutput> GetParam(const ::String& clientName, const ::String& paramName) = 0;
    virtual ara::core::Future<MonitorParamOutput> MonitorParam(const ::String& clientName, const ::String& paramName) = 0;
    virtual ara::core::Future<SetParamOutput> SetParam(const ::String& clientName, const ::String& paramName, const ::String& paramValue, const ::UInt8& paramType, const ::UInt8& persistType) = 0;
    virtual ara::core::Future<UnMonitorParamOutput> UnMonitorParam(const ::String& clientName, const ::String& paramName) = 0;
    virtual ara::core::Future<InitClientOutput> InitClient(const ::String& clientName) = 0;
    virtual void E2EErrorHandler(ara::com::e2e::E2EErrorCode, ara::com::e2e::DataID, ara::com::e2e::MessageCounter){}

    events::ParamUpdateEvent ParamUpdateEvent;
    events::ServerNotifyEvent ServerNotifyEvent;
    methods::AnswerAliveHandle AnswerAliveHandle;
    methods::DelParamHandle DelParamHandle;
    methods::GetMonitorClientsHandle GetMonitorClientsHandle;
    methods::GetParamHandle GetParamHandle;
    methods::MonitorParamHandle MonitorParamHandle;
    methods::SetParamHandle SetParamHandle;
    methods::UnMonitorParamHandle UnMonitorParamHandle;
    methods::InitClientHandle InitClientHandle;
};
} // namespace skeleton
} // namespace server
} // namespace config
} // namespace mdc

#endif // MDC_CONFIG_SERVER_CONFIGSERVERSERVICEINTERFACE_SKELETON_H
