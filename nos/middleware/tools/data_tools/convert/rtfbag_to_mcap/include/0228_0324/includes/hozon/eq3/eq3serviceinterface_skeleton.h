/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_EQ3_EQ3SERVICEINTERFACE_SKELETON_H
#define HOZON_EQ3_EQ3SERVICEINTERFACE_SKELETON_H

#include "ara/com/internal/skeleton/skeleton_adapter.h"
#include "ara/com/internal/skeleton/event_adapter.h"
#include "ara/com/internal/skeleton/field_adapter.h"
#include "ara/com/internal/skeleton/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/eq3/eq3serviceinterface_common.h"
#include <cstdint>

namespace hozon {
namespace eq3 {
namespace skeleton {
namespace events
{
    using Eq3VisDataEvent = ara::com::internal::skeleton::event::EventAdapter<::hozon::eq3::Eq3VisDataType>;
    using Eq3PedestrianDataEvent = ara::com::internal::skeleton::event::EventAdapter<::hozon::eq3::PedestrianInfos>;
    using Eq3RtdisDataEvent = ara::com::internal::skeleton::event::EventAdapter<::hozon::eq3::RTDisInfos>;
    using Eq3RtsdisDataEvent = ara::com::internal::skeleton::event::EventAdapter<::hozon::eq3::RTSDisInfos>;
    using Eq3VisObsMsgEvent = ara::com::internal::skeleton::event::EventAdapter<::hozon::eq3::VisObsMsgsDataType>;
    static constexpr ara::com::internal::EntityId Eq3ServiceInterfaceEq3VisDataEventId = 48405U; //Eq3VisDataEvent_event_hash
    static constexpr ara::com::internal::EntityId Eq3ServiceInterfaceEq3PedestrianDataEventId = 10191U; //Eq3PedestrianDataEvent_event_hash
    static constexpr ara::com::internal::EntityId Eq3ServiceInterfaceEq3RtdisDataEventId = 34878U; //Eq3RtdisDataEvent_event_hash
    static constexpr ara::com::internal::EntityId Eq3ServiceInterfaceEq3RtsdisDataEventId = 7435U; //Eq3RtsdisDataEvent_event_hash
    static constexpr ara::com::internal::EntityId Eq3ServiceInterfaceEq3VisObsMsgEventId = 3675U; //Eq3VisObsMsgEvent_event_hash
}

namespace methods
{
}

namespace fields
{
}

class Eq3ServiceInterfaceSkeleton {
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
        const ara::core::Result<void> resultEq3VisDataEvent = skeletonAdapter->InitializeEvent(Eq3VisDataEvent);
        ThrowError(resultEq3VisDataEvent);
        const ara::core::Result<void> resultEq3PedestrianDataEvent = skeletonAdapter->InitializeEvent(Eq3PedestrianDataEvent);
        ThrowError(resultEq3PedestrianDataEvent);
        const ara::core::Result<void> resultEq3RtdisDataEvent = skeletonAdapter->InitializeEvent(Eq3RtdisDataEvent);
        ThrowError(resultEq3RtdisDataEvent);
        const ara::core::Result<void> resultEq3RtsdisDataEvent = skeletonAdapter->InitializeEvent(Eq3RtsdisDataEvent);
        ThrowError(resultEq3RtsdisDataEvent);
        const ara::core::Result<void> resultEq3VisObsMsgEvent = skeletonAdapter->InitializeEvent(Eq3VisObsMsgEvent);
        ThrowError(resultEq3VisObsMsgEvent);
    }

    Eq3ServiceInterfaceSkeleton& operator=(const Eq3ServiceInterfaceSkeleton&) = delete;

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
    explicit Eq3ServiceInterfaceSkeleton(const ara::com::InstanceIdentifier& instanceId,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        : skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::eq3::Eq3ServiceInterface::ServiceIdentifier, instanceId, mode)),
          Eq3VisDataEvent(skeletonAdapter->GetSkeleton(), events::Eq3ServiceInterfaceEq3VisDataEventId),
          Eq3PedestrianDataEvent(skeletonAdapter->GetSkeleton(), events::Eq3ServiceInterfaceEq3PedestrianDataEventId),
          Eq3RtdisDataEvent(skeletonAdapter->GetSkeleton(), events::Eq3ServiceInterfaceEq3RtdisDataEventId),
          Eq3RtsdisDataEvent(skeletonAdapter->GetSkeleton(), events::Eq3ServiceInterfaceEq3RtsdisDataEventId),
          Eq3VisObsMsgEvent(skeletonAdapter->GetSkeleton(), events::Eq3ServiceInterfaceEq3VisObsMsgEventId){
        ConstructSkeleton(mode);
    }

    explicit Eq3ServiceInterfaceSkeleton(const ara::core::InstanceSpecifier& instanceSpec,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::eq3::Eq3ServiceInterface::ServiceIdentifier, instanceSpec, mode)),
          Eq3VisDataEvent(skeletonAdapter->GetSkeleton(), events::Eq3ServiceInterfaceEq3VisDataEventId),
          Eq3PedestrianDataEvent(skeletonAdapter->GetSkeleton(), events::Eq3ServiceInterfaceEq3PedestrianDataEventId),
          Eq3RtdisDataEvent(skeletonAdapter->GetSkeleton(), events::Eq3ServiceInterfaceEq3RtdisDataEventId),
          Eq3RtsdisDataEvent(skeletonAdapter->GetSkeleton(), events::Eq3ServiceInterfaceEq3RtsdisDataEventId),
          Eq3VisObsMsgEvent(skeletonAdapter->GetSkeleton(), events::Eq3ServiceInterfaceEq3VisObsMsgEventId){
        ConstructSkeleton(mode);
    }

    explicit Eq3ServiceInterfaceSkeleton(const ara::com::InstanceIdentifierContainer instanceContainer,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::eq3::Eq3ServiceInterface::ServiceIdentifier, instanceContainer, mode)),
          Eq3VisDataEvent(skeletonAdapter->GetSkeleton(), events::Eq3ServiceInterfaceEq3VisDataEventId),
          Eq3PedestrianDataEvent(skeletonAdapter->GetSkeleton(), events::Eq3ServiceInterfaceEq3PedestrianDataEventId),
          Eq3RtdisDataEvent(skeletonAdapter->GetSkeleton(), events::Eq3ServiceInterfaceEq3RtdisDataEventId),
          Eq3RtsdisDataEvent(skeletonAdapter->GetSkeleton(), events::Eq3ServiceInterfaceEq3RtsdisDataEventId),
          Eq3VisObsMsgEvent(skeletonAdapter->GetSkeleton(), events::Eq3ServiceInterfaceEq3VisObsMsgEventId){
        ConstructSkeleton(mode);
    }

    Eq3ServiceInterfaceSkeleton(const Eq3ServiceInterfaceSkeleton&) = delete;

    Eq3ServiceInterfaceSkeleton(Eq3ServiceInterfaceSkeleton&&) = default;
    Eq3ServiceInterfaceSkeleton& operator=(Eq3ServiceInterfaceSkeleton&&) = default;
    Eq3ServiceInterfaceSkeleton(ConstructionToken&& token) noexcept 
        : skeletonAdapter(token.GetSkeletonAdapter()),
          Eq3VisDataEvent(skeletonAdapter->GetSkeleton(), events::Eq3ServiceInterfaceEq3VisDataEventId),
          Eq3PedestrianDataEvent(skeletonAdapter->GetSkeleton(), events::Eq3ServiceInterfaceEq3PedestrianDataEventId),
          Eq3RtdisDataEvent(skeletonAdapter->GetSkeleton(), events::Eq3ServiceInterfaceEq3RtdisDataEventId),
          Eq3RtsdisDataEvent(skeletonAdapter->GetSkeleton(), events::Eq3ServiceInterfaceEq3RtsdisDataEventId),
          Eq3VisObsMsgEvent(skeletonAdapter->GetSkeleton(), events::Eq3ServiceInterfaceEq3VisObsMsgEventId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifier instanceId,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::eq3::Eq3ServiceInterface::ServiceIdentifier, instanceId, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::core::InstanceSpecifier instanceSpec,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::eq3::Eq3ServiceInterface::ServiceIdentifier, instanceSpec, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifierContainer instanceIdContainer,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::eq3::Eq3ServiceInterface::ServiceIdentifier, instanceIdContainer, mode);
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
            const events::Eq3VisDataEvent Eq3VisDataEvent(preSkeletonAdapter->GetSkeleton(), events::Eq3ServiceInterfaceEq3VisDataEventId);
            initResult = preSkeletonAdapter->InitializeEvent(Eq3VisDataEvent);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const events::Eq3PedestrianDataEvent Eq3PedestrianDataEvent(preSkeletonAdapter->GetSkeleton(), events::Eq3ServiceInterfaceEq3PedestrianDataEventId);
            initResult = preSkeletonAdapter->InitializeEvent(Eq3PedestrianDataEvent);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const events::Eq3RtdisDataEvent Eq3RtdisDataEvent(preSkeletonAdapter->GetSkeleton(), events::Eq3ServiceInterfaceEq3RtdisDataEventId);
            initResult = preSkeletonAdapter->InitializeEvent(Eq3RtdisDataEvent);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const events::Eq3RtsdisDataEvent Eq3RtsdisDataEvent(preSkeletonAdapter->GetSkeleton(), events::Eq3ServiceInterfaceEq3RtsdisDataEventId);
            initResult = preSkeletonAdapter->InitializeEvent(Eq3RtsdisDataEvent);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const events::Eq3VisObsMsgEvent Eq3VisObsMsgEvent(preSkeletonAdapter->GetSkeleton(), events::Eq3ServiceInterfaceEq3VisObsMsgEventId);
            initResult = preSkeletonAdapter->InitializeEvent(Eq3VisObsMsgEvent);
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

    virtual ~Eq3ServiceInterfaceSkeleton()
    {
        StopOfferService();
    }

    void OfferService()
    {
        skeletonAdapter->RegisterE2EErrorHandler(&Eq3ServiceInterfaceSkeleton::E2EErrorHandler, *this);
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

    events::Eq3VisDataEvent Eq3VisDataEvent;
    events::Eq3PedestrianDataEvent Eq3PedestrianDataEvent;
    events::Eq3RtdisDataEvent Eq3RtdisDataEvent;
    events::Eq3RtsdisDataEvent Eq3RtsdisDataEvent;
    events::Eq3VisObsMsgEvent Eq3VisObsMsgEvent;
};
} // namespace skeleton
} // namespace eq3
} // namespace hozon

#endif // HOZON_EQ3_EQ3SERVICEINTERFACE_SKELETON_H
