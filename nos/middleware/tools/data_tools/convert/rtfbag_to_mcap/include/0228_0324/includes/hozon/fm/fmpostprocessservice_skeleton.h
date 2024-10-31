/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_FM_FMPOSTPROCESSSERVICE_SKELETON_H
#define HOZON_FM_FMPOSTPROCESSSERVICE_SKELETON_H

#include "ara/com/internal/skeleton/skeleton_adapter.h"
#include "ara/com/internal/skeleton/event_adapter.h"
#include "ara/com/internal/skeleton/field_adapter.h"
#include "ara/com/internal/skeleton/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/fm/fmpostprocessservice_common.h"
#include <cstdint>

namespace hozon {
namespace fm {
namespace skeleton {
namespace events
{
    using FaultPostProcessEvent = ara::com::internal::skeleton::event::EventAdapter<::hozon::fm::FaultClusterData>;
    static constexpr ara::com::internal::EntityId FmPostProcessServiceFaultPostProcessEventId = 26600U; //FaultPostProcessEvent_event_hash
}

namespace methods
{
    using RegistPostProcessFaultHandle = ara::com::internal::skeleton::method::MethodAdapter;
    static constexpr ara::com::internal::EntityId FmPostProcessServiceRegistPostProcessFaultId = 34209U; //RegistPostProcessFault_method_hash
}

namespace fields
{
}

class FmPostProcessServiceSkeleton {
private:
    std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> skeletonAdapter;
    void ConstructSkeleton(const ara::com::MethodCallProcessingMode mode)
    {
        if (mode == ara::com::MethodCallProcessingMode::kEvent) {
            if (!(skeletonAdapter->SetMethodThreadNumber(skeletonAdapter->GetMethodThreadNumber(1U), 1024U))) {
#ifndef NOT_SUPPORT_EXCEPTIONS
                ara::core::ErrorCode errorcode(ara::com::ComErrc::kNetworkBindingFailure);
                throw ara::com::ComException(std::move(errorcode));
#else
                std::cerr << "Error: Not support exception, create skeleton failed!"<< std::endl;
#endif
            }
        }
        const ara::core::Result<void> resultFaultPostProcessEvent = skeletonAdapter->InitializeEvent(FaultPostProcessEvent);
        ThrowError(resultFaultPostProcessEvent);
        const ara::core::Result<void> resultRegistPostProcessFault = skeletonAdapter->InitializeMethod<RegistPostProcessFaultOutput>(methods::FmPostProcessServiceRegistPostProcessFaultId);
        ThrowError(resultRegistPostProcessFault);
    }

    FmPostProcessServiceSkeleton& operator=(const FmPostProcessServiceSkeleton&) = delete;

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
    using RegistPostProcessFaultOutput = void;
    
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
    explicit FmPostProcessServiceSkeleton(const ara::com::InstanceIdentifier& instanceId,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        : skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::fm::FmPostProcessService::ServiceIdentifier, instanceId, mode)),
          FaultPostProcessEvent(skeletonAdapter->GetSkeleton(), events::FmPostProcessServiceFaultPostProcessEventId),
          RegistPostProcessFaultHandle(skeletonAdapter->GetSkeleton(), methods::FmPostProcessServiceRegistPostProcessFaultId){
        ConstructSkeleton(mode);
    }

    explicit FmPostProcessServiceSkeleton(const ara::core::InstanceSpecifier& instanceSpec,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::fm::FmPostProcessService::ServiceIdentifier, instanceSpec, mode)),
          FaultPostProcessEvent(skeletonAdapter->GetSkeleton(), events::FmPostProcessServiceFaultPostProcessEventId),
          RegistPostProcessFaultHandle(skeletonAdapter->GetSkeleton(), methods::FmPostProcessServiceRegistPostProcessFaultId){
        ConstructSkeleton(mode);
    }

    explicit FmPostProcessServiceSkeleton(const ara::com::InstanceIdentifierContainer instanceContainer,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::fm::FmPostProcessService::ServiceIdentifier, instanceContainer, mode)),
          FaultPostProcessEvent(skeletonAdapter->GetSkeleton(), events::FmPostProcessServiceFaultPostProcessEventId),
          RegistPostProcessFaultHandle(skeletonAdapter->GetSkeleton(), methods::FmPostProcessServiceRegistPostProcessFaultId){
        ConstructSkeleton(mode);
    }

    FmPostProcessServiceSkeleton(const FmPostProcessServiceSkeleton&) = delete;

    FmPostProcessServiceSkeleton(FmPostProcessServiceSkeleton&&) = default;
    FmPostProcessServiceSkeleton& operator=(FmPostProcessServiceSkeleton&&) = default;
    FmPostProcessServiceSkeleton(ConstructionToken&& token) noexcept 
        : skeletonAdapter(token.GetSkeletonAdapter()),
          FaultPostProcessEvent(skeletonAdapter->GetSkeleton(), events::FmPostProcessServiceFaultPostProcessEventId),
          RegistPostProcessFaultHandle(skeletonAdapter->GetSkeleton(), methods::FmPostProcessServiceRegistPostProcessFaultId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifier instanceId,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::fm::FmPostProcessService::ServiceIdentifier, instanceId, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::core::InstanceSpecifier instanceSpec,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::fm::FmPostProcessService::ServiceIdentifier, instanceSpec, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifierContainer instanceIdContainer,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::fm::FmPostProcessService::ServiceIdentifier, instanceIdContainer, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> PreConstructResult(
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter>& preSkeletonAdapter, const ara::com::MethodCallProcessingMode mode)
    {
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            if (mode == ara::com::MethodCallProcessingMode::kEvent) {
                if(!preSkeletonAdapter->SetMethodThreadNumber(preSkeletonAdapter->GetMethodThreadNumber(1U), 1024U)) {
                    result = false;
                    ara::core::ErrorCode errorcode(ara::com::ComErrc::kNetworkBindingFailure);
                    initResult.EmplaceError(errorcode);
                    break;
                }
            }
            const events::FaultPostProcessEvent FaultPostProcessEvent(preSkeletonAdapter->GetSkeleton(), events::FmPostProcessServiceFaultPostProcessEventId);
            initResult = preSkeletonAdapter->InitializeEvent(FaultPostProcessEvent);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<RegistPostProcessFaultOutput>(methods::FmPostProcessServiceRegistPostProcessFaultId);
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

    virtual ~FmPostProcessServiceSkeleton()
    {
        StopOfferService();
    }

    void OfferService()
    {
        skeletonAdapter->RegisterE2EErrorHandler(&FmPostProcessServiceSkeleton::E2EErrorHandler, *this);
        skeletonAdapter->RegisterMethod(&FmPostProcessServiceSkeleton::RegistPostProcessFault, *this, methods::FmPostProcessServiceRegistPostProcessFaultId);
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
    virtual RegistPostProcessFaultOutput RegistPostProcessFault(const ::String& appName, const ::StringVector& clusterList) = 0;
    virtual void E2EErrorHandler(ara::com::e2e::E2EErrorCode, ara::com::e2e::DataID, ara::com::e2e::MessageCounter){}

    events::FaultPostProcessEvent FaultPostProcessEvent;
    methods::RegistPostProcessFaultHandle RegistPostProcessFaultHandle;
};
} // namespace skeleton
} // namespace fm
} // namespace hozon

#endif // HOZON_FM_FMPOSTPROCESSSERVICE_SKELETON_H
