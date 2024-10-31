/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_LPM_SOCPOWERSERVICEINTERFACE_SKELETON_H
#define HOZON_LPM_SOCPOWERSERVICEINTERFACE_SKELETON_H

#include "ara/com/internal/skeleton/skeleton_adapter.h"
#include "ara/com/internal/skeleton/event_adapter.h"
#include "ara/com/internal/skeleton/field_adapter.h"
#include "ara/com/internal/skeleton/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/lpm/socpowerserviceinterface_common.h"
#include <cstdint>

namespace hozon {
namespace lpm {
namespace skeleton {
namespace events
{
    using LowPowerConditionEvent = ara::com::internal::skeleton::event::EventAdapter<::Int8>;
    static constexpr ara::com::internal::EntityId SocPowerServiceInterfaceLowPowerConditionEventId = 56894U; //LowPowerConditionEvent_event_hash
}

namespace methods
{
    using RequestLowPowerHandle = ara::com::internal::skeleton::method::MethodAdapter;
    static constexpr ara::com::internal::EntityId SocPowerServiceInterfaceRequestLowPowerId = 60221U; //RequestLowPower_method_hash
}

namespace fields
{
}

class SocPowerServiceInterfaceSkeleton {
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
        const ara::core::Result<void> resultLowPowerConditionEvent = skeletonAdapter->InitializeEvent(LowPowerConditionEvent);
        ThrowError(resultLowPowerConditionEvent);
        const ara::core::Result<void> resultRequestLowPower = skeletonAdapter->InitializeMethod<ara::core::Future<RequestLowPowerOutput>>(methods::SocPowerServiceInterfaceRequestLowPowerId);
        ThrowError(resultRequestLowPower);
    }

    SocPowerServiceInterfaceSkeleton& operator=(const SocPowerServiceInterfaceSkeleton&) = delete;

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
    using RequestLowPowerOutput = hozon::lpm::methods::RequestLowPower::Output;
    
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
    explicit SocPowerServiceInterfaceSkeleton(const ara::com::InstanceIdentifier& instanceId,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        : skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::lpm::SocPowerServiceInterface::ServiceIdentifier, instanceId, mode)),
          LowPowerConditionEvent(skeletonAdapter->GetSkeleton(), events::SocPowerServiceInterfaceLowPowerConditionEventId),
          RequestLowPowerHandle(skeletonAdapter->GetSkeleton(), methods::SocPowerServiceInterfaceRequestLowPowerId){
        ConstructSkeleton(mode);
    }

    explicit SocPowerServiceInterfaceSkeleton(const ara::core::InstanceSpecifier& instanceSpec,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::lpm::SocPowerServiceInterface::ServiceIdentifier, instanceSpec, mode)),
          LowPowerConditionEvent(skeletonAdapter->GetSkeleton(), events::SocPowerServiceInterfaceLowPowerConditionEventId),
          RequestLowPowerHandle(skeletonAdapter->GetSkeleton(), methods::SocPowerServiceInterfaceRequestLowPowerId){
        ConstructSkeleton(mode);
    }

    explicit SocPowerServiceInterfaceSkeleton(const ara::com::InstanceIdentifierContainer instanceContainer,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::lpm::SocPowerServiceInterface::ServiceIdentifier, instanceContainer, mode)),
          LowPowerConditionEvent(skeletonAdapter->GetSkeleton(), events::SocPowerServiceInterfaceLowPowerConditionEventId),
          RequestLowPowerHandle(skeletonAdapter->GetSkeleton(), methods::SocPowerServiceInterfaceRequestLowPowerId){
        ConstructSkeleton(mode);
    }

    SocPowerServiceInterfaceSkeleton(const SocPowerServiceInterfaceSkeleton&) = delete;

    SocPowerServiceInterfaceSkeleton(SocPowerServiceInterfaceSkeleton&&) = default;
    SocPowerServiceInterfaceSkeleton& operator=(SocPowerServiceInterfaceSkeleton&&) = default;
    SocPowerServiceInterfaceSkeleton(ConstructionToken&& token) noexcept 
        : skeletonAdapter(token.GetSkeletonAdapter()),
          LowPowerConditionEvent(skeletonAdapter->GetSkeleton(), events::SocPowerServiceInterfaceLowPowerConditionEventId),
          RequestLowPowerHandle(skeletonAdapter->GetSkeleton(), methods::SocPowerServiceInterfaceRequestLowPowerId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifier instanceId,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::lpm::SocPowerServiceInterface::ServiceIdentifier, instanceId, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::core::InstanceSpecifier instanceSpec,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::lpm::SocPowerServiceInterface::ServiceIdentifier, instanceSpec, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifierContainer instanceIdContainer,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::lpm::SocPowerServiceInterface::ServiceIdentifier, instanceIdContainer, mode);
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
            const events::LowPowerConditionEvent LowPowerConditionEvent(preSkeletonAdapter->GetSkeleton(), events::SocPowerServiceInterfaceLowPowerConditionEventId);
            initResult = preSkeletonAdapter->InitializeEvent(LowPowerConditionEvent);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<RequestLowPowerOutput>>(methods::SocPowerServiceInterfaceRequestLowPowerId);
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

    virtual ~SocPowerServiceInterfaceSkeleton()
    {
        StopOfferService();
    }

    void OfferService()
    {
        skeletonAdapter->RegisterE2EErrorHandler(&SocPowerServiceInterfaceSkeleton::E2EErrorHandler, *this);
        skeletonAdapter->RegisterMethod(&SocPowerServiceInterfaceSkeleton::RequestLowPower, *this, methods::SocPowerServiceInterfaceRequestLowPowerId);
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
    virtual ara::core::Future<RequestLowPowerOutput> RequestLowPower(const ::hozon::lpm::LpmMcuRequest& lowPowerRequest) = 0;
    virtual void E2EErrorHandler(ara::com::e2e::E2EErrorCode, ara::com::e2e::DataID, ara::com::e2e::MessageCounter){}

    events::LowPowerConditionEvent LowPowerConditionEvent;
    methods::RequestLowPowerHandle RequestLowPowerHandle;
};
} // namespace skeleton
} // namespace lpm
} // namespace hozon

#endif // HOZON_LPM_SOCPOWERSERVICEINTERFACE_SKELETON_H
