/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_LOWSPEEDBSD_HOZONINTERFACE_LOW_SPD_BSD_SKELETON_H
#define HOZON_INTERFACE_LOWSPEEDBSD_HOZONINTERFACE_LOW_SPD_BSD_SKELETON_H

#include "ara/com/internal/skeleton/skeleton_adapter.h"
#include "ara/com/internal/skeleton/event_adapter.h"
#include "ara/com/internal/skeleton/field_adapter.h"
#include "ara/com/internal/skeleton/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/interface/lowspeedbsd/hozoninterface_low_spd_bsd_common.h"
#include <cstdint>

namespace hozon {
namespace interface {
namespace lowspeedbsd {
namespace skeleton {
namespace events
{
    using hozonLowSpdBsdEvent = ara::com::internal::skeleton::event::EventAdapter<::hozon::lowspeedbsd::LowSpeedBsdSignal>;
    static constexpr ara::com::internal::EntityId HozonInterface_low_spd_bsdhozonLowSpdBsdEventId = 1160U; //hozonLowSpdBsdEvent_event_hash
}

namespace methods
{
}

namespace fields
{
}

class HozonInterface_low_spd_bsdSkeleton {
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
        const ara::core::Result<void> resulthozonLowSpdBsdEvent = skeletonAdapter->InitializeEvent(hozonLowSpdBsdEvent);
        ThrowError(resulthozonLowSpdBsdEvent);
    }

    HozonInterface_low_spd_bsdSkeleton& operator=(const HozonInterface_low_spd_bsdSkeleton&) = delete;

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
    explicit HozonInterface_low_spd_bsdSkeleton(const ara::com::InstanceIdentifier& instanceId,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        : skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::interface::lowspeedbsd::HozonInterface_low_spd_bsd::ServiceIdentifier, instanceId, mode)),
          hozonLowSpdBsdEvent(skeletonAdapter->GetSkeleton(), events::HozonInterface_low_spd_bsdhozonLowSpdBsdEventId){
        ConstructSkeleton(mode);
    }

    explicit HozonInterface_low_spd_bsdSkeleton(const ara::core::InstanceSpecifier& instanceSpec,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::interface::lowspeedbsd::HozonInterface_low_spd_bsd::ServiceIdentifier, instanceSpec, mode)),
          hozonLowSpdBsdEvent(skeletonAdapter->GetSkeleton(), events::HozonInterface_low_spd_bsdhozonLowSpdBsdEventId){
        ConstructSkeleton(mode);
    }

    explicit HozonInterface_low_spd_bsdSkeleton(const ara::com::InstanceIdentifierContainer instanceContainer,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::interface::lowspeedbsd::HozonInterface_low_spd_bsd::ServiceIdentifier, instanceContainer, mode)),
          hozonLowSpdBsdEvent(skeletonAdapter->GetSkeleton(), events::HozonInterface_low_spd_bsdhozonLowSpdBsdEventId){
        ConstructSkeleton(mode);
    }

    HozonInterface_low_spd_bsdSkeleton(const HozonInterface_low_spd_bsdSkeleton&) = delete;

    HozonInterface_low_spd_bsdSkeleton(HozonInterface_low_spd_bsdSkeleton&&) = default;
    HozonInterface_low_spd_bsdSkeleton& operator=(HozonInterface_low_spd_bsdSkeleton&&) = default;
    HozonInterface_low_spd_bsdSkeleton(ConstructionToken&& token) noexcept 
        : skeletonAdapter(token.GetSkeletonAdapter()),
          hozonLowSpdBsdEvent(skeletonAdapter->GetSkeleton(), events::HozonInterface_low_spd_bsdhozonLowSpdBsdEventId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifier instanceId,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::interface::lowspeedbsd::HozonInterface_low_spd_bsd::ServiceIdentifier, instanceId, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::core::InstanceSpecifier instanceSpec,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::interface::lowspeedbsd::HozonInterface_low_spd_bsd::ServiceIdentifier, instanceSpec, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifierContainer instanceIdContainer,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::interface::lowspeedbsd::HozonInterface_low_spd_bsd::ServiceIdentifier, instanceIdContainer, mode);
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
            const events::hozonLowSpdBsdEvent hozonLowSpdBsdEvent(preSkeletonAdapter->GetSkeleton(), events::HozonInterface_low_spd_bsdhozonLowSpdBsdEventId);
            initResult = preSkeletonAdapter->InitializeEvent(hozonLowSpdBsdEvent);
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

    virtual ~HozonInterface_low_spd_bsdSkeleton()
    {
        StopOfferService();
    }

    void OfferService()
    {
        skeletonAdapter->RegisterE2EErrorHandler(&HozonInterface_low_spd_bsdSkeleton::E2EErrorHandler, *this);
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

    events::hozonLowSpdBsdEvent hozonLowSpdBsdEvent;
};
} // namespace skeleton
} // namespace lowspeedbsd
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_LOWSPEEDBSD_HOZONINTERFACE_LOW_SPD_BSD_SKELETON_H