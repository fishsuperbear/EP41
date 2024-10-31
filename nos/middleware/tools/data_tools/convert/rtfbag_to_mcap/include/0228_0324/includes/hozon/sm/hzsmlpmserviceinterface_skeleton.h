/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SM_HZSMLPMSERVICEINTERFACE_SKELETON_H
#define HOZON_SM_HZSMLPMSERVICEINTERFACE_SKELETON_H

#include "ara/com/internal/skeleton/skeleton_adapter.h"
#include "ara/com/internal/skeleton/event_adapter.h"
#include "ara/com/internal/skeleton/field_adapter.h"
#include "ara/com/internal/skeleton/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/sm/hzsmlpmserviceinterface_common.h"
#include <cstdint>

namespace hozon {
namespace sm {
namespace skeleton {
namespace events
{
    using SmLpmDdsEvent = ara::com::internal::skeleton::event::EventAdapter<::int8_t>;
    static constexpr ara::com::internal::EntityId HzSmLpmServiceInterfaceSmLpmDdsEventId = 39120U; //SmLpmDdsEvent_event_hash
}

namespace methods
{
}

namespace fields
{
}

class HzSmLpmServiceInterfaceSkeleton {
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
        const ara::core::Result<void> resultSmLpmDdsEvent = skeletonAdapter->InitializeEvent(SmLpmDdsEvent);
        ThrowError(resultSmLpmDdsEvent);
    }

    HzSmLpmServiceInterfaceSkeleton& operator=(const HzSmLpmServiceInterfaceSkeleton&) = delete;

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
    explicit HzSmLpmServiceInterfaceSkeleton(const ara::com::InstanceIdentifier& instanceId,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        : skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::sm::HzSmLpmServiceInterface::ServiceIdentifier, instanceId, mode)),
          SmLpmDdsEvent(skeletonAdapter->GetSkeleton(), events::HzSmLpmServiceInterfaceSmLpmDdsEventId){
        ConstructSkeleton(mode);
    }

    explicit HzSmLpmServiceInterfaceSkeleton(const ara::core::InstanceSpecifier& instanceSpec,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::sm::HzSmLpmServiceInterface::ServiceIdentifier, instanceSpec, mode)),
          SmLpmDdsEvent(skeletonAdapter->GetSkeleton(), events::HzSmLpmServiceInterfaceSmLpmDdsEventId){
        ConstructSkeleton(mode);
    }

    explicit HzSmLpmServiceInterfaceSkeleton(const ara::com::InstanceIdentifierContainer instanceContainer,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::sm::HzSmLpmServiceInterface::ServiceIdentifier, instanceContainer, mode)),
          SmLpmDdsEvent(skeletonAdapter->GetSkeleton(), events::HzSmLpmServiceInterfaceSmLpmDdsEventId){
        ConstructSkeleton(mode);
    }

    HzSmLpmServiceInterfaceSkeleton(const HzSmLpmServiceInterfaceSkeleton&) = delete;

    HzSmLpmServiceInterfaceSkeleton(HzSmLpmServiceInterfaceSkeleton&&) = default;
    HzSmLpmServiceInterfaceSkeleton& operator=(HzSmLpmServiceInterfaceSkeleton&&) = default;
    HzSmLpmServiceInterfaceSkeleton(ConstructionToken&& token) noexcept 
        : skeletonAdapter(token.GetSkeletonAdapter()),
          SmLpmDdsEvent(skeletonAdapter->GetSkeleton(), events::HzSmLpmServiceInterfaceSmLpmDdsEventId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifier instanceId,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::sm::HzSmLpmServiceInterface::ServiceIdentifier, instanceId, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::core::InstanceSpecifier instanceSpec,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::sm::HzSmLpmServiceInterface::ServiceIdentifier, instanceSpec, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifierContainer instanceIdContainer,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::sm::HzSmLpmServiceInterface::ServiceIdentifier, instanceIdContainer, mode);
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
            const events::SmLpmDdsEvent SmLpmDdsEvent(preSkeletonAdapter->GetSkeleton(), events::HzSmLpmServiceInterfaceSmLpmDdsEventId);
            initResult = preSkeletonAdapter->InitializeEvent(SmLpmDdsEvent);
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

    virtual ~HzSmLpmServiceInterfaceSkeleton()
    {
        StopOfferService();
    }

    void OfferService()
    {
        skeletonAdapter->RegisterE2EErrorHandler(&HzSmLpmServiceInterfaceSkeleton::E2EErrorHandler, *this);
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

    events::SmLpmDdsEvent SmLpmDdsEvent;
};
} // namespace skeleton
} // namespace sm
} // namespace hozon

#endif // HOZON_SM_HZSMLPMSERVICEINTERFACE_SKELETON_H
