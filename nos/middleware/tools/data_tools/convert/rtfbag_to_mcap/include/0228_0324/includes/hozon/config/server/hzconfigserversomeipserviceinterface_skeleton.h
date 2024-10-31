/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CONFIG_SERVER_HZCONFIGSERVERSOMEIPSERVICEINTERFACE_SKELETON_H
#define HOZON_CONFIG_SERVER_HZCONFIGSERVERSOMEIPSERVICEINTERFACE_SKELETON_H

#include "ara/com/internal/skeleton/skeleton_adapter.h"
#include "ara/com/internal/skeleton/event_adapter.h"
#include "ara/com/internal/skeleton/field_adapter.h"
#include "ara/com/internal/skeleton/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/config/server/hzconfigserversomeipserviceinterface_common.h"
#include <cstdint>

namespace hozon {
namespace config {
namespace server {
namespace skeleton {
namespace events
{
    using HzParamUpdateEvent = ara::com::internal::skeleton::event::EventAdapter<::hozon::config::server::HzParamUpdateData>;
    using HzServerNotifyEvent = ara::com::internal::skeleton::event::EventAdapter<::hozon::config::server::HzServerNotifyData>;
    using VehicleCfgUpdateToMcuEvent = ara::com::internal::skeleton::event::EventAdapter<::hozon::config::server::struct_config_array>;
    static constexpr ara::com::internal::EntityId HzConfigServerSomeipServiceInterfaceHzParamUpdateEventId = 29525U; //HzParamUpdateEvent_event_hash
    static constexpr ara::com::internal::EntityId HzConfigServerSomeipServiceInterfaceHzServerNotifyEventId = 3223U; //HzServerNotifyEvent_event_hash
    static constexpr ara::com::internal::EntityId HzConfigServerSomeipServiceInterfaceVehicleCfgUpdateToMcuEventId = 5435U; //VehicleCfgUpdateToMcuEvent_event_hash
}

namespace methods
{
    using HzAnswerAliveHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using HzDelParamHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using HzGetMonitorClientsHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using HzGetParamHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using HzMonitorParamHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using HzSetParamHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using HzUnMonitorParamHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using VehicleCfgUpdateResFromMcuHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using HzGetVehicleCfgParamHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using HzGetVINCfgParamHandle = ara::com::internal::skeleton::method::MethodAdapter;
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
}

namespace fields
{
}

class HzConfigServerSomeipServiceInterfaceSkeleton {
private:
    std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> skeletonAdapter;
    void ConstructSkeleton(const ara::com::MethodCallProcessingMode mode)
    {
        if (mode == ara::com::MethodCallProcessingMode::kEvent) {
            if (!(skeletonAdapter->SetMethodThreadNumber(skeletonAdapter->GetMethodThreadNumber(10U), 1024U))) {
#ifndef NOT_SUPPORT_EXCEPTIONS
                ara::core::ErrorCode errorcode(ara::com::ComErrc::kNetworkBindingFailure);
                throw ara::com::ComException(std::move(errorcode));
#else
                std::cerr << "Error: Not support exception, create skeleton failed!"<< std::endl;
#endif
            }
        }
        const ara::core::Result<void> resultHzParamUpdateEvent = skeletonAdapter->InitializeEvent(HzParamUpdateEvent);
        ThrowError(resultHzParamUpdateEvent);
        const ara::core::Result<void> resultHzServerNotifyEvent = skeletonAdapter->InitializeEvent(HzServerNotifyEvent);
        ThrowError(resultHzServerNotifyEvent);
        const ara::core::Result<void> resultVehicleCfgUpdateToMcuEvent = skeletonAdapter->InitializeEvent(VehicleCfgUpdateToMcuEvent);
        ThrowError(resultVehicleCfgUpdateToMcuEvent);
        const ara::core::Result<void> resultHzAnswerAlive = skeletonAdapter->InitializeMethod<ara::core::Future<HzAnswerAliveOutput>>(methods::HzConfigServerSomeipServiceInterfaceHzAnswerAliveId);
        ThrowError(resultHzAnswerAlive);
        const ara::core::Result<void> resultHzDelParam = skeletonAdapter->InitializeMethod<ara::core::Future<HzDelParamOutput>>(methods::HzConfigServerSomeipServiceInterfaceHzDelParamId);
        ThrowError(resultHzDelParam);
        const ara::core::Result<void> resultHzGetMonitorClients = skeletonAdapter->InitializeMethod<ara::core::Future<HzGetMonitorClientsOutput>>(methods::HzConfigServerSomeipServiceInterfaceHzGetMonitorClientsId);
        ThrowError(resultHzGetMonitorClients);
        const ara::core::Result<void> resultHzGetParam = skeletonAdapter->InitializeMethod<ara::core::Future<HzGetParamOutput>>(methods::HzConfigServerSomeipServiceInterfaceHzGetParamId);
        ThrowError(resultHzGetParam);
        const ara::core::Result<void> resultHzMonitorParam = skeletonAdapter->InitializeMethod<ara::core::Future<HzMonitorParamOutput>>(methods::HzConfigServerSomeipServiceInterfaceHzMonitorParamId);
        ThrowError(resultHzMonitorParam);
        const ara::core::Result<void> resultHzSetParam = skeletonAdapter->InitializeMethod<ara::core::Future<HzSetParamOutput>>(methods::HzConfigServerSomeipServiceInterfaceHzSetParamId);
        ThrowError(resultHzSetParam);
        const ara::core::Result<void> resultHzUnMonitorParam = skeletonAdapter->InitializeMethod<ara::core::Future<HzUnMonitorParamOutput>>(methods::HzConfigServerSomeipServiceInterfaceHzUnMonitorParamId);
        ThrowError(resultHzUnMonitorParam);
        const ara::core::Result<void> resultVehicleCfgUpdateResFromMcu = skeletonAdapter->InitializeMethod<ara::core::Future<VehicleCfgUpdateResFromMcuOutput>>(methods::HzConfigServerSomeipServiceInterfaceVehicleCfgUpdateResFromMcuId);
        ThrowError(resultVehicleCfgUpdateResFromMcu);
        const ara::core::Result<void> resultHzGetVehicleCfgParam = skeletonAdapter->InitializeMethod<ara::core::Future<HzGetVehicleCfgParamOutput>>(methods::HzConfigServerSomeipServiceInterfaceHzGetVehicleCfgParamId);
        ThrowError(resultHzGetVehicleCfgParam);
        const ara::core::Result<void> resultHzGetVINCfgParam = skeletonAdapter->InitializeMethod<ara::core::Future<HzGetVINCfgParamOutput>>(methods::HzConfigServerSomeipServiceInterfaceHzGetVINCfgParamId);
        ThrowError(resultHzGetVINCfgParam);
    }

    HzConfigServerSomeipServiceInterfaceSkeleton& operator=(const HzConfigServerSomeipServiceInterfaceSkeleton&) = delete;

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
    using HzAnswerAliveOutput = hozon::config::server::methods::HzAnswerAlive::Output;
    
    using HzDelParamOutput = hozon::config::server::methods::HzDelParam::Output;
    
    using HzGetMonitorClientsOutput = hozon::config::server::methods::HzGetMonitorClients::Output;
    
    using HzGetParamOutput = hozon::config::server::methods::HzGetParam::Output;
    
    using HzMonitorParamOutput = hozon::config::server::methods::HzMonitorParam::Output;
    
    using HzSetParamOutput = hozon::config::server::methods::HzSetParam::Output;
    
    using HzUnMonitorParamOutput = hozon::config::server::methods::HzUnMonitorParam::Output;
    
    using VehicleCfgUpdateResFromMcuOutput = hozon::config::server::methods::VehicleCfgUpdateResFromMcu::Output;
    
    using HzGetVehicleCfgParamOutput = hozon::config::server::methods::HzGetVehicleCfgParam::Output;
    
    using HzGetVINCfgParamOutput = hozon::config::server::methods::HzGetVINCfgParam::Output;
    
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
    explicit HzConfigServerSomeipServiceInterfaceSkeleton(const ara::com::InstanceIdentifier& instanceId,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        : skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::config::server::HzConfigServerSomeipServiceInterface::ServiceIdentifier, instanceId, mode)),
          HzParamUpdateEvent(skeletonAdapter->GetSkeleton(), events::HzConfigServerSomeipServiceInterfaceHzParamUpdateEventId),
          HzServerNotifyEvent(skeletonAdapter->GetSkeleton(), events::HzConfigServerSomeipServiceInterfaceHzServerNotifyEventId),
          VehicleCfgUpdateToMcuEvent(skeletonAdapter->GetSkeleton(), events::HzConfigServerSomeipServiceInterfaceVehicleCfgUpdateToMcuEventId),
          HzAnswerAliveHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzAnswerAliveId),
          HzDelParamHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzDelParamId),
          HzGetMonitorClientsHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzGetMonitorClientsId),
          HzGetParamHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzGetParamId),
          HzMonitorParamHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzMonitorParamId),
          HzSetParamHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzSetParamId),
          HzUnMonitorParamHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzUnMonitorParamId),
          VehicleCfgUpdateResFromMcuHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceVehicleCfgUpdateResFromMcuId),
          HzGetVehicleCfgParamHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzGetVehicleCfgParamId),
          HzGetVINCfgParamHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzGetVINCfgParamId){
        ConstructSkeleton(mode);
    }

    explicit HzConfigServerSomeipServiceInterfaceSkeleton(const ara::core::InstanceSpecifier& instanceSpec,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::config::server::HzConfigServerSomeipServiceInterface::ServiceIdentifier, instanceSpec, mode)),
          HzParamUpdateEvent(skeletonAdapter->GetSkeleton(), events::HzConfigServerSomeipServiceInterfaceHzParamUpdateEventId),
          HzServerNotifyEvent(skeletonAdapter->GetSkeleton(), events::HzConfigServerSomeipServiceInterfaceHzServerNotifyEventId),
          VehicleCfgUpdateToMcuEvent(skeletonAdapter->GetSkeleton(), events::HzConfigServerSomeipServiceInterfaceVehicleCfgUpdateToMcuEventId),
          HzAnswerAliveHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzAnswerAliveId),
          HzDelParamHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzDelParamId),
          HzGetMonitorClientsHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzGetMonitorClientsId),
          HzGetParamHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzGetParamId),
          HzMonitorParamHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzMonitorParamId),
          HzSetParamHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzSetParamId),
          HzUnMonitorParamHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzUnMonitorParamId),
          VehicleCfgUpdateResFromMcuHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceVehicleCfgUpdateResFromMcuId),
          HzGetVehicleCfgParamHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzGetVehicleCfgParamId),
          HzGetVINCfgParamHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzGetVINCfgParamId){
        ConstructSkeleton(mode);
    }

    explicit HzConfigServerSomeipServiceInterfaceSkeleton(const ara::com::InstanceIdentifierContainer instanceContainer,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::config::server::HzConfigServerSomeipServiceInterface::ServiceIdentifier, instanceContainer, mode)),
          HzParamUpdateEvent(skeletonAdapter->GetSkeleton(), events::HzConfigServerSomeipServiceInterfaceHzParamUpdateEventId),
          HzServerNotifyEvent(skeletonAdapter->GetSkeleton(), events::HzConfigServerSomeipServiceInterfaceHzServerNotifyEventId),
          VehicleCfgUpdateToMcuEvent(skeletonAdapter->GetSkeleton(), events::HzConfigServerSomeipServiceInterfaceVehicleCfgUpdateToMcuEventId),
          HzAnswerAliveHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzAnswerAliveId),
          HzDelParamHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzDelParamId),
          HzGetMonitorClientsHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzGetMonitorClientsId),
          HzGetParamHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzGetParamId),
          HzMonitorParamHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzMonitorParamId),
          HzSetParamHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzSetParamId),
          HzUnMonitorParamHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzUnMonitorParamId),
          VehicleCfgUpdateResFromMcuHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceVehicleCfgUpdateResFromMcuId),
          HzGetVehicleCfgParamHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzGetVehicleCfgParamId),
          HzGetVINCfgParamHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzGetVINCfgParamId){
        ConstructSkeleton(mode);
    }

    HzConfigServerSomeipServiceInterfaceSkeleton(const HzConfigServerSomeipServiceInterfaceSkeleton&) = delete;

    HzConfigServerSomeipServiceInterfaceSkeleton(HzConfigServerSomeipServiceInterfaceSkeleton&&) = default;
    HzConfigServerSomeipServiceInterfaceSkeleton& operator=(HzConfigServerSomeipServiceInterfaceSkeleton&&) = default;
    HzConfigServerSomeipServiceInterfaceSkeleton(ConstructionToken&& token) noexcept 
        : skeletonAdapter(token.GetSkeletonAdapter()),
          HzParamUpdateEvent(skeletonAdapter->GetSkeleton(), events::HzConfigServerSomeipServiceInterfaceHzParamUpdateEventId),
          HzServerNotifyEvent(skeletonAdapter->GetSkeleton(), events::HzConfigServerSomeipServiceInterfaceHzServerNotifyEventId),
          VehicleCfgUpdateToMcuEvent(skeletonAdapter->GetSkeleton(), events::HzConfigServerSomeipServiceInterfaceVehicleCfgUpdateToMcuEventId),
          HzAnswerAliveHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzAnswerAliveId),
          HzDelParamHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzDelParamId),
          HzGetMonitorClientsHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzGetMonitorClientsId),
          HzGetParamHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzGetParamId),
          HzMonitorParamHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzMonitorParamId),
          HzSetParamHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzSetParamId),
          HzUnMonitorParamHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzUnMonitorParamId),
          VehicleCfgUpdateResFromMcuHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceVehicleCfgUpdateResFromMcuId),
          HzGetVehicleCfgParamHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzGetVehicleCfgParamId),
          HzGetVINCfgParamHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerSomeipServiceInterfaceHzGetVINCfgParamId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifier instanceId,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::config::server::HzConfigServerSomeipServiceInterface::ServiceIdentifier, instanceId, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::core::InstanceSpecifier instanceSpec,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::config::server::HzConfigServerSomeipServiceInterface::ServiceIdentifier, instanceSpec, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifierContainer instanceIdContainer,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::config::server::HzConfigServerSomeipServiceInterface::ServiceIdentifier, instanceIdContainer, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> PreConstructResult(
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter>& preSkeletonAdapter, const ara::com::MethodCallProcessingMode mode)
    {
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            if (mode == ara::com::MethodCallProcessingMode::kEvent) {
                if(!preSkeletonAdapter->SetMethodThreadNumber(preSkeletonAdapter->GetMethodThreadNumber(10U), 1024U)) {
                    result = false;
                    ara::core::ErrorCode errorcode(ara::com::ComErrc::kNetworkBindingFailure);
                    initResult.EmplaceError(errorcode);
                    break;
                }
            }
            const events::HzParamUpdateEvent HzParamUpdateEvent(preSkeletonAdapter->GetSkeleton(), events::HzConfigServerSomeipServiceInterfaceHzParamUpdateEventId);
            initResult = preSkeletonAdapter->InitializeEvent(HzParamUpdateEvent);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const events::HzServerNotifyEvent HzServerNotifyEvent(preSkeletonAdapter->GetSkeleton(), events::HzConfigServerSomeipServiceInterfaceHzServerNotifyEventId);
            initResult = preSkeletonAdapter->InitializeEvent(HzServerNotifyEvent);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const events::VehicleCfgUpdateToMcuEvent VehicleCfgUpdateToMcuEvent(preSkeletonAdapter->GetSkeleton(), events::HzConfigServerSomeipServiceInterfaceVehicleCfgUpdateToMcuEventId);
            initResult = preSkeletonAdapter->InitializeEvent(VehicleCfgUpdateToMcuEvent);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<HzAnswerAliveOutput>>(methods::HzConfigServerSomeipServiceInterfaceHzAnswerAliveId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<HzDelParamOutput>>(methods::HzConfigServerSomeipServiceInterfaceHzDelParamId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<HzGetMonitorClientsOutput>>(methods::HzConfigServerSomeipServiceInterfaceHzGetMonitorClientsId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<HzGetParamOutput>>(methods::HzConfigServerSomeipServiceInterfaceHzGetParamId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<HzMonitorParamOutput>>(methods::HzConfigServerSomeipServiceInterfaceHzMonitorParamId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<HzSetParamOutput>>(methods::HzConfigServerSomeipServiceInterfaceHzSetParamId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<HzUnMonitorParamOutput>>(methods::HzConfigServerSomeipServiceInterfaceHzUnMonitorParamId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<VehicleCfgUpdateResFromMcuOutput>>(methods::HzConfigServerSomeipServiceInterfaceVehicleCfgUpdateResFromMcuId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<HzGetVehicleCfgParamOutput>>(methods::HzConfigServerSomeipServiceInterfaceHzGetVehicleCfgParamId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<HzGetVINCfgParamOutput>>(methods::HzConfigServerSomeipServiceInterfaceHzGetVINCfgParamId);
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

    virtual ~HzConfigServerSomeipServiceInterfaceSkeleton()
    {
        StopOfferService();
    }

    void OfferService()
    {
        skeletonAdapter->RegisterE2EErrorHandler(&HzConfigServerSomeipServiceInterfaceSkeleton::E2EErrorHandler, *this);
        skeletonAdapter->RegisterMethod(&HzConfigServerSomeipServiceInterfaceSkeleton::HzAnswerAlive, *this, methods::HzConfigServerSomeipServiceInterfaceHzAnswerAliveId);
        skeletonAdapter->RegisterMethod(&HzConfigServerSomeipServiceInterfaceSkeleton::HzDelParam, *this, methods::HzConfigServerSomeipServiceInterfaceHzDelParamId);
        skeletonAdapter->RegisterMethod(&HzConfigServerSomeipServiceInterfaceSkeleton::HzGetMonitorClients, *this, methods::HzConfigServerSomeipServiceInterfaceHzGetMonitorClientsId);
        skeletonAdapter->RegisterMethod(&HzConfigServerSomeipServiceInterfaceSkeleton::HzGetParam, *this, methods::HzConfigServerSomeipServiceInterfaceHzGetParamId);
        skeletonAdapter->RegisterMethod(&HzConfigServerSomeipServiceInterfaceSkeleton::HzMonitorParam, *this, methods::HzConfigServerSomeipServiceInterfaceHzMonitorParamId);
        skeletonAdapter->RegisterMethod(&HzConfigServerSomeipServiceInterfaceSkeleton::HzSetParam, *this, methods::HzConfigServerSomeipServiceInterfaceHzSetParamId);
        skeletonAdapter->RegisterMethod(&HzConfigServerSomeipServiceInterfaceSkeleton::HzUnMonitorParam, *this, methods::HzConfigServerSomeipServiceInterfaceHzUnMonitorParamId);
        skeletonAdapter->RegisterMethod(&HzConfigServerSomeipServiceInterfaceSkeleton::VehicleCfgUpdateResFromMcu, *this, methods::HzConfigServerSomeipServiceInterfaceVehicleCfgUpdateResFromMcuId);
        skeletonAdapter->RegisterMethod(&HzConfigServerSomeipServiceInterfaceSkeleton::HzGetVehicleCfgParam, *this, methods::HzConfigServerSomeipServiceInterfaceHzGetVehicleCfgParamId);
        skeletonAdapter->RegisterMethod(&HzConfigServerSomeipServiceInterfaceSkeleton::HzGetVINCfgParam, *this, methods::HzConfigServerSomeipServiceInterfaceHzGetVINCfgParamId);
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
    virtual ara::core::Future<HzAnswerAliveOutput> HzAnswerAlive(const ::String& clientName) = 0;
    virtual ara::core::Future<HzDelParamOutput> HzDelParam(const ::String& clientName, const ::String& paramName) = 0;
    virtual ara::core::Future<HzGetMonitorClientsOutput> HzGetMonitorClients(const ::String& clientName, const ::String& paramName) = 0;
    virtual ara::core::Future<HzGetParamOutput> HzGetParam(const ::String& clientName, const ::String& paramName, const ::UInt8& paramTypeIn) = 0;
    virtual ara::core::Future<HzMonitorParamOutput> HzMonitorParam(const ::String& clientName, const ::String& paramName) = 0;
    virtual ara::core::Future<HzSetParamOutput> HzSetParam(const ::String& clientName, const ::String& paramName, const ::String& paramValue, const ::UInt8& paramType, const ::Boolean& isPersist) = 0;
    virtual ara::core::Future<HzUnMonitorParamOutput> HzUnMonitorParam(const ::String& clientName, const ::String& paramName) = 0;
    virtual ara::core::Future<VehicleCfgUpdateResFromMcuOutput> VehicleCfgUpdateResFromMcu(const ::UInt8& returnCode) = 0;
    virtual ara::core::Future<HzGetVehicleCfgParamOutput> HzGetVehicleCfgParam(const ::String& ModuleName, const ::String& paramName) = 0;
    virtual ara::core::Future<HzGetVINCfgParamOutput> HzGetVINCfgParam(const ::String& ModuleName, const ::String& paramName) = 0;
    virtual void E2EErrorHandler(ara::com::e2e::E2EErrorCode, ara::com::e2e::DataID, ara::com::e2e::MessageCounter){}

    events::HzParamUpdateEvent HzParamUpdateEvent;
    events::HzServerNotifyEvent HzServerNotifyEvent;
    events::VehicleCfgUpdateToMcuEvent VehicleCfgUpdateToMcuEvent;
    methods::HzAnswerAliveHandle HzAnswerAliveHandle;
    methods::HzDelParamHandle HzDelParamHandle;
    methods::HzGetMonitorClientsHandle HzGetMonitorClientsHandle;
    methods::HzGetParamHandle HzGetParamHandle;
    methods::HzMonitorParamHandle HzMonitorParamHandle;
    methods::HzSetParamHandle HzSetParamHandle;
    methods::HzUnMonitorParamHandle HzUnMonitorParamHandle;
    methods::VehicleCfgUpdateResFromMcuHandle VehicleCfgUpdateResFromMcuHandle;
    methods::HzGetVehicleCfgParamHandle HzGetVehicleCfgParamHandle;
    methods::HzGetVINCfgParamHandle HzGetVINCfgParamHandle;
};
} // namespace skeleton
} // namespace server
} // namespace config
} // namespace hozon

#endif // HOZON_CONFIG_SERVER_HZCONFIGSERVERSOMEIPSERVICEINTERFACE_SKELETON_H
