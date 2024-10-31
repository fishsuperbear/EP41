/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_VEHICLESERVICEINTERFACE_SKELETON_H
#define ARA_VEHICLE_VEHICLESERVICEINTERFACE_SKELETON_H

#include "ara/com/internal/skeleton/skeleton_adapter.h"
#include "ara/com/internal/skeleton/event_adapter.h"
#include "ara/com/internal/skeleton/field_adapter.h"
#include "ara/com/internal/skeleton/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "ara/vehicle/vehicleserviceinterface_common.h"
#include <cstdint>

namespace ara {
namespace vehicle {
namespace skeleton {
namespace events
{
    using VehicleDataEvent = ara::com::internal::skeleton::event::EventAdapter<::ara::vehicle::VehicleInfo>;
    using FLCInfoEvent = ara::com::internal::skeleton::event::EventAdapter<::ara::vehicle::FLCInfo>;
    using FLRInfoEvent = ara::com::internal::skeleton::event::EventAdapter<::ara::vehicle::FLRInfo>;
    static constexpr ara::com::internal::EntityId VehicleServiceInterfaceVehicleDataEventId = 30140U; //VehicleDataEvent_event_hash
    static constexpr ara::com::internal::EntityId VehicleServiceInterfaceFLCInfoEventId = 41206U; //FLCInfoEvent_event_hash
    static constexpr ara::com::internal::EntityId VehicleServiceInterfaceFLRInfoEventId = 32763U; //FLRInfoEvent_event_hash
}

namespace methods
{
}

namespace fields
{
}

class VehicleServiceInterfaceSkeleton {
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
        const ara::core::Result<void> resultVehicleDataEvent = skeletonAdapter->InitializeEvent(VehicleDataEvent);
        ThrowError(resultVehicleDataEvent);
        const ara::core::Result<void> resultFLCInfoEvent = skeletonAdapter->InitializeEvent(FLCInfoEvent);
        ThrowError(resultFLCInfoEvent);
        const ara::core::Result<void> resultFLRInfoEvent = skeletonAdapter->InitializeEvent(FLRInfoEvent);
        ThrowError(resultFLRInfoEvent);
    }

    VehicleServiceInterfaceSkeleton& operator=(const VehicleServiceInterfaceSkeleton&) = delete;

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
    explicit VehicleServiceInterfaceSkeleton(const ara::com::InstanceIdentifier& instanceId,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        : skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::ara::vehicle::VehicleServiceInterface::ServiceIdentifier, instanceId, mode)),
          VehicleDataEvent(skeletonAdapter->GetSkeleton(), events::VehicleServiceInterfaceVehicleDataEventId),
          FLCInfoEvent(skeletonAdapter->GetSkeleton(), events::VehicleServiceInterfaceFLCInfoEventId),
          FLRInfoEvent(skeletonAdapter->GetSkeleton(), events::VehicleServiceInterfaceFLRInfoEventId){
        ConstructSkeleton(mode);
    }

    explicit VehicleServiceInterfaceSkeleton(const ara::core::InstanceSpecifier& instanceSpec,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::ara::vehicle::VehicleServiceInterface::ServiceIdentifier, instanceSpec, mode)),
          VehicleDataEvent(skeletonAdapter->GetSkeleton(), events::VehicleServiceInterfaceVehicleDataEventId),
          FLCInfoEvent(skeletonAdapter->GetSkeleton(), events::VehicleServiceInterfaceFLCInfoEventId),
          FLRInfoEvent(skeletonAdapter->GetSkeleton(), events::VehicleServiceInterfaceFLRInfoEventId){
        ConstructSkeleton(mode);
    }

    explicit VehicleServiceInterfaceSkeleton(const ara::com::InstanceIdentifierContainer instanceContainer,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::ara::vehicle::VehicleServiceInterface::ServiceIdentifier, instanceContainer, mode)),
          VehicleDataEvent(skeletonAdapter->GetSkeleton(), events::VehicleServiceInterfaceVehicleDataEventId),
          FLCInfoEvent(skeletonAdapter->GetSkeleton(), events::VehicleServiceInterfaceFLCInfoEventId),
          FLRInfoEvent(skeletonAdapter->GetSkeleton(), events::VehicleServiceInterfaceFLRInfoEventId){
        ConstructSkeleton(mode);
    }

    VehicleServiceInterfaceSkeleton(const VehicleServiceInterfaceSkeleton&) = delete;

    VehicleServiceInterfaceSkeleton(VehicleServiceInterfaceSkeleton&&) = default;
    VehicleServiceInterfaceSkeleton& operator=(VehicleServiceInterfaceSkeleton&&) = default;
    VehicleServiceInterfaceSkeleton(ConstructionToken&& token) noexcept 
        : skeletonAdapter(token.GetSkeletonAdapter()),
          VehicleDataEvent(skeletonAdapter->GetSkeleton(), events::VehicleServiceInterfaceVehicleDataEventId),
          FLCInfoEvent(skeletonAdapter->GetSkeleton(), events::VehicleServiceInterfaceFLCInfoEventId),
          FLRInfoEvent(skeletonAdapter->GetSkeleton(), events::VehicleServiceInterfaceFLRInfoEventId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifier instanceId,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::ara::vehicle::VehicleServiceInterface::ServiceIdentifier, instanceId, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::core::InstanceSpecifier instanceSpec,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::ara::vehicle::VehicleServiceInterface::ServiceIdentifier, instanceSpec, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifierContainer instanceIdContainer,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::ara::vehicle::VehicleServiceInterface::ServiceIdentifier, instanceIdContainer, mode);
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
            const events::VehicleDataEvent VehicleDataEvent(preSkeletonAdapter->GetSkeleton(), events::VehicleServiceInterfaceVehicleDataEventId);
            initResult = preSkeletonAdapter->InitializeEvent(VehicleDataEvent);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const events::FLCInfoEvent FLCInfoEvent(preSkeletonAdapter->GetSkeleton(), events::VehicleServiceInterfaceFLCInfoEventId);
            initResult = preSkeletonAdapter->InitializeEvent(FLCInfoEvent);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const events::FLRInfoEvent FLRInfoEvent(preSkeletonAdapter->GetSkeleton(), events::VehicleServiceInterfaceFLRInfoEventId);
            initResult = preSkeletonAdapter->InitializeEvent(FLRInfoEvent);
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

    virtual ~VehicleServiceInterfaceSkeleton()
    {
        StopOfferService();
    }

    void OfferService()
    {
        skeletonAdapter->RegisterE2EErrorHandler(&VehicleServiceInterfaceSkeleton::E2EErrorHandler, *this);
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

    events::VehicleDataEvent VehicleDataEvent;
    events::FLCInfoEvent FLCInfoEvent;
    events::FLRInfoEvent FLRInfoEvent;
};
} // namespace skeleton
} // namespace vehicle
} // namespace ara

#endif // ARA_VEHICLE_VEHICLESERVICEINTERFACE_SKELETON_H
