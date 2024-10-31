/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_ADAS_ADASSERVICEINTERFACE_SKELETON_H
#define ARA_ADAS_ADASSERVICEINTERFACE_SKELETON_H

#include "ara/com/internal/skeleton/skeleton_adapter.h"
#include "ara/com/internal/skeleton/event_adapter.h"
#include "ara/com/internal/skeleton/field_adapter.h"
#include "ara/com/internal/skeleton/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "ara/adas/adasserviceinterface_common.h"
#include <cstdint>

namespace ara {
namespace adas {
namespace skeleton {
namespace events
{
    using FLCFr01InfoEvent = ara::com::internal::skeleton::event::EventAdapter<::ara::vehicle::FLCFr01Info>;
    using FLCFr02InfoEvent = ara::com::internal::skeleton::event::EventAdapter<::ara::vehicle::FLCFr02Info>;
    using FLRFr01InfoEvent = ara::com::internal::skeleton::event::EventAdapter<::ara::vehicle::FLRFr01Info>;
    using FLRFr02InfoEvent = ara::com::internal::skeleton::event::EventAdapter<::ara::vehicle::FLRFr02Info>;
    using FLRFr03InfoEvent = ara::com::internal::skeleton::event::EventAdapter<::ara::vehicle::FLRFr03Info>;
    using ApaInfoEvent = ara::com::internal::skeleton::event::EventAdapter<::ara::vehicle::APAInfo>;
    static constexpr ara::com::internal::EntityId AdasServiceInterfaceFLCFr01InfoEventId = 60499U; //FLCFr01InfoEvent_event_hash
    static constexpr ara::com::internal::EntityId AdasServiceInterfaceFLCFr02InfoEventId = 17391U; //FLCFr02InfoEvent_event_hash
    static constexpr ara::com::internal::EntityId AdasServiceInterfaceFLRFr01InfoEventId = 30176U; //FLRFr01InfoEvent_event_hash
    static constexpr ara::com::internal::EntityId AdasServiceInterfaceFLRFr02InfoEventId = 2444U; //FLRFr02InfoEvent_event_hash
    static constexpr ara::com::internal::EntityId AdasServiceInterfaceFLRFr03InfoEventId = 38387U; //FLRFr03InfoEvent_event_hash
    static constexpr ara::com::internal::EntityId AdasServiceInterfaceApaInfoEventId = 52677U; //ApaInfoEvent_event_hash
}

namespace methods
{
}

namespace fields
{
}

class AdasServiceInterfaceSkeleton {
private:
    std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> skeletonAdapter;
    void ConstructSkeleton(const ara::com::MethodCallProcessingMode mode)
    {
        if (mode == ara::com::MethodCallProcessingMode::kEvent) {
            if (!(skeletonAdapter->SetMethodThreadNumber(skeletonAdapter->GetMethodThreadNumber(0U), 1024U))) {
#ifndef NOT_SUPPORT_EXCEPTIONS
                ara::core::ErrorCode errorcode(ara::com::ComErrc::kNetworkBindingFailure);
                throw ara::com::ComException(std::move(errorcode));
#else
                std::cerr << "Error: Not support exception, create skeleton failed!"<< std::endl;
#endif
            }
        }
        const ara::core::Result<void> resultFLCFr01InfoEvent = skeletonAdapter->InitializeEvent(FLCFr01InfoEvent);
        ThrowError(resultFLCFr01InfoEvent);
        const ara::core::Result<void> resultFLCFr02InfoEvent = skeletonAdapter->InitializeEvent(FLCFr02InfoEvent);
        ThrowError(resultFLCFr02InfoEvent);
        const ara::core::Result<void> resultFLRFr01InfoEvent = skeletonAdapter->InitializeEvent(FLRFr01InfoEvent);
        ThrowError(resultFLRFr01InfoEvent);
        const ara::core::Result<void> resultFLRFr02InfoEvent = skeletonAdapter->InitializeEvent(FLRFr02InfoEvent);
        ThrowError(resultFLRFr02InfoEvent);
        const ara::core::Result<void> resultFLRFr03InfoEvent = skeletonAdapter->InitializeEvent(FLRFr03InfoEvent);
        ThrowError(resultFLRFr03InfoEvent);
        const ara::core::Result<void> resultApaInfoEvent = skeletonAdapter->InitializeEvent(ApaInfoEvent);
        ThrowError(resultApaInfoEvent);
    }

    AdasServiceInterfaceSkeleton& operator=(const AdasServiceInterfaceSkeleton&) = delete;

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
    explicit AdasServiceInterfaceSkeleton(const ara::com::InstanceIdentifier& instanceId,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        : skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::ara::adas::AdasServiceInterface::ServiceIdentifier, instanceId, mode)),
          FLCFr01InfoEvent(skeletonAdapter->GetSkeleton(), events::AdasServiceInterfaceFLCFr01InfoEventId),
          FLCFr02InfoEvent(skeletonAdapter->GetSkeleton(), events::AdasServiceInterfaceFLCFr02InfoEventId),
          FLRFr01InfoEvent(skeletonAdapter->GetSkeleton(), events::AdasServiceInterfaceFLRFr01InfoEventId),
          FLRFr02InfoEvent(skeletonAdapter->GetSkeleton(), events::AdasServiceInterfaceFLRFr02InfoEventId),
          FLRFr03InfoEvent(skeletonAdapter->GetSkeleton(), events::AdasServiceInterfaceFLRFr03InfoEventId),
          ApaInfoEvent(skeletonAdapter->GetSkeleton(), events::AdasServiceInterfaceApaInfoEventId){
        ConstructSkeleton(mode);
    }

    explicit AdasServiceInterfaceSkeleton(const ara::core::InstanceSpecifier& instanceSpec,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::ara::adas::AdasServiceInterface::ServiceIdentifier, instanceSpec, mode)),
          FLCFr01InfoEvent(skeletonAdapter->GetSkeleton(), events::AdasServiceInterfaceFLCFr01InfoEventId),
          FLCFr02InfoEvent(skeletonAdapter->GetSkeleton(), events::AdasServiceInterfaceFLCFr02InfoEventId),
          FLRFr01InfoEvent(skeletonAdapter->GetSkeleton(), events::AdasServiceInterfaceFLRFr01InfoEventId),
          FLRFr02InfoEvent(skeletonAdapter->GetSkeleton(), events::AdasServiceInterfaceFLRFr02InfoEventId),
          FLRFr03InfoEvent(skeletonAdapter->GetSkeleton(), events::AdasServiceInterfaceFLRFr03InfoEventId),
          ApaInfoEvent(skeletonAdapter->GetSkeleton(), events::AdasServiceInterfaceApaInfoEventId){
        ConstructSkeleton(mode);
    }

    explicit AdasServiceInterfaceSkeleton(const ara::com::InstanceIdentifierContainer instanceContainer,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::ara::adas::AdasServiceInterface::ServiceIdentifier, instanceContainer, mode)),
          FLCFr01InfoEvent(skeletonAdapter->GetSkeleton(), events::AdasServiceInterfaceFLCFr01InfoEventId),
          FLCFr02InfoEvent(skeletonAdapter->GetSkeleton(), events::AdasServiceInterfaceFLCFr02InfoEventId),
          FLRFr01InfoEvent(skeletonAdapter->GetSkeleton(), events::AdasServiceInterfaceFLRFr01InfoEventId),
          FLRFr02InfoEvent(skeletonAdapter->GetSkeleton(), events::AdasServiceInterfaceFLRFr02InfoEventId),
          FLRFr03InfoEvent(skeletonAdapter->GetSkeleton(), events::AdasServiceInterfaceFLRFr03InfoEventId),
          ApaInfoEvent(skeletonAdapter->GetSkeleton(), events::AdasServiceInterfaceApaInfoEventId){
        ConstructSkeleton(mode);
    }

    AdasServiceInterfaceSkeleton(const AdasServiceInterfaceSkeleton&) = delete;

    AdasServiceInterfaceSkeleton(AdasServiceInterfaceSkeleton&&) = default;
    AdasServiceInterfaceSkeleton& operator=(AdasServiceInterfaceSkeleton&&) = default;
    AdasServiceInterfaceSkeleton(ConstructionToken&& token) noexcept 
        : skeletonAdapter(token.GetSkeletonAdapter()),
          FLCFr01InfoEvent(skeletonAdapter->GetSkeleton(), events::AdasServiceInterfaceFLCFr01InfoEventId),
          FLCFr02InfoEvent(skeletonAdapter->GetSkeleton(), events::AdasServiceInterfaceFLCFr02InfoEventId),
          FLRFr01InfoEvent(skeletonAdapter->GetSkeleton(), events::AdasServiceInterfaceFLRFr01InfoEventId),
          FLRFr02InfoEvent(skeletonAdapter->GetSkeleton(), events::AdasServiceInterfaceFLRFr02InfoEventId),
          FLRFr03InfoEvent(skeletonAdapter->GetSkeleton(), events::AdasServiceInterfaceFLRFr03InfoEventId),
          ApaInfoEvent(skeletonAdapter->GetSkeleton(), events::AdasServiceInterfaceApaInfoEventId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifier instanceId,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::ara::adas::AdasServiceInterface::ServiceIdentifier, instanceId, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::core::InstanceSpecifier instanceSpec,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::ara::adas::AdasServiceInterface::ServiceIdentifier, instanceSpec, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifierContainer instanceIdContainer,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::ara::adas::AdasServiceInterface::ServiceIdentifier, instanceIdContainer, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> PreConstructResult(
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter>& preSkeletonAdapter, const ara::com::MethodCallProcessingMode mode)
    {
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            if (mode == ara::com::MethodCallProcessingMode::kEvent) {
                if(!preSkeletonAdapter->SetMethodThreadNumber(preSkeletonAdapter->GetMethodThreadNumber(0U), 1024U)) {
                    result = false;
                    ara::core::ErrorCode errorcode(ara::com::ComErrc::kNetworkBindingFailure);
                    initResult.EmplaceError(errorcode);
                    break;
                }
            }
            const events::FLCFr01InfoEvent FLCFr01InfoEvent(preSkeletonAdapter->GetSkeleton(), events::AdasServiceInterfaceFLCFr01InfoEventId);
            initResult = preSkeletonAdapter->InitializeEvent(FLCFr01InfoEvent);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const events::FLCFr02InfoEvent FLCFr02InfoEvent(preSkeletonAdapter->GetSkeleton(), events::AdasServiceInterfaceFLCFr02InfoEventId);
            initResult = preSkeletonAdapter->InitializeEvent(FLCFr02InfoEvent);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const events::FLRFr01InfoEvent FLRFr01InfoEvent(preSkeletonAdapter->GetSkeleton(), events::AdasServiceInterfaceFLRFr01InfoEventId);
            initResult = preSkeletonAdapter->InitializeEvent(FLRFr01InfoEvent);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const events::FLRFr02InfoEvent FLRFr02InfoEvent(preSkeletonAdapter->GetSkeleton(), events::AdasServiceInterfaceFLRFr02InfoEventId);
            initResult = preSkeletonAdapter->InitializeEvent(FLRFr02InfoEvent);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const events::FLRFr03InfoEvent FLRFr03InfoEvent(preSkeletonAdapter->GetSkeleton(), events::AdasServiceInterfaceFLRFr03InfoEventId);
            initResult = preSkeletonAdapter->InitializeEvent(FLRFr03InfoEvent);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const events::ApaInfoEvent ApaInfoEvent(preSkeletonAdapter->GetSkeleton(), events::AdasServiceInterfaceApaInfoEventId);
            initResult = preSkeletonAdapter->InitializeEvent(ApaInfoEvent);
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

    virtual ~AdasServiceInterfaceSkeleton()
    {
        StopOfferService();
    }

    void OfferService()
    {
        skeletonAdapter->RegisterE2EErrorHandler(&AdasServiceInterfaceSkeleton::E2EErrorHandler, *this);
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
    virtual void E2EErrorHandler(ara::com::e2e::E2EErrorCode, ara::com::e2e::DataID, ara::com::e2e::MessageCounter){}

    events::FLCFr01InfoEvent FLCFr01InfoEvent;
    events::FLCFr02InfoEvent FLCFr02InfoEvent;
    events::FLRFr01InfoEvent FLRFr01InfoEvent;
    events::FLRFr02InfoEvent FLRFr02InfoEvent;
    events::FLRFr03InfoEvent FLRFr03InfoEvent;
    events::ApaInfoEvent ApaInfoEvent;
};
} // namespace skeleton
} // namespace adas
} // namespace ara

#endif // ARA_ADAS_ADASSERVICEINTERFACE_SKELETON_H
