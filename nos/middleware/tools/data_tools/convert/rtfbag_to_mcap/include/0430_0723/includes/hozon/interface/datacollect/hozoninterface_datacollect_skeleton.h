/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_DATACOLLECT_HOZONINTERFACE_DATACOLLECT_SKELETON_H
#define HOZON_INTERFACE_DATACOLLECT_HOZONINTERFACE_DATACOLLECT_SKELETON_H

#include "ara/com/internal/skeleton/skeleton_adapter.h"
#include "ara/com/internal/skeleton/event_adapter.h"
#include "ara/com/internal/skeleton/field_adapter.h"
#include "ara/com/internal/skeleton/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/interface/datacollect/hozoninterface_datacollect_common.h"
#include <cstdint>

namespace hozon {
namespace interface {
namespace datacollect {
namespace skeleton {
namespace events
{
}

namespace methods
{
    using CollectTriggerReqHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using CollectCustomDataReqHandle = ara::com::internal::skeleton::method::MethodAdapter;
    static constexpr ara::com::internal::EntityId HozonInterface_DataCollectCollectTriggerReqId = 28344U; //CollectTriggerReq_method_hash
    static constexpr ara::com::internal::EntityId HozonInterface_DataCollectCollectCustomDataReqId = 41558U; //CollectCustomDataReq_method_hash
}

namespace fields
{
}

class HozonInterface_DataCollectSkeleton {
private:
    std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> skeletonAdapter;
    void ConstructSkeleton(const ara::com::MethodCallProcessingMode mode)
    {
        if (mode == ara::com::MethodCallProcessingMode::kEvent) {
            if (!(skeletonAdapter->SetMethodThreadNumber(skeletonAdapter->GetMethodThreadNumber(2U), 1024U))) {
#ifndef NOT_SUPPORT_EXCEPTIONS
                ara::core::ErrorCode errorcode(ara::com::ComErrc::kNetworkBindingFailure);
                throw ara::com::ComException(std::move(errorcode));
#else
                std::cerr << "Error: Not support exception, create skeleton failed!"<< std::endl;
#endif
            }
        }
        const ara::core::Result<void> resultCollectTriggerReq = skeletonAdapter->InitializeMethod<CollectTriggerReqOutput>(methods::HozonInterface_DataCollectCollectTriggerReqId);
        ThrowError(resultCollectTriggerReq);
        const ara::core::Result<void> resultCollectCustomDataReq = skeletonAdapter->InitializeMethod<CollectCustomDataReqOutput>(methods::HozonInterface_DataCollectCollectCustomDataReqId);
        ThrowError(resultCollectCustomDataReq);
    }

    HozonInterface_DataCollectSkeleton& operator=(const HozonInterface_DataCollectSkeleton&) = delete;

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
    using CollectTriggerReqOutput = void;
    
    using CollectCustomDataReqOutput = void;
    
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
    explicit HozonInterface_DataCollectSkeleton(const ara::com::InstanceIdentifier& instanceId,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        : skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::interface::datacollect::HozonInterface_DataCollect::ServiceIdentifier, instanceId, mode)),
          CollectTriggerReqHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_DataCollectCollectTriggerReqId),
          CollectCustomDataReqHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_DataCollectCollectCustomDataReqId){
        ConstructSkeleton(mode);
    }

    explicit HozonInterface_DataCollectSkeleton(const ara::core::InstanceSpecifier& instanceSpec,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::interface::datacollect::HozonInterface_DataCollect::ServiceIdentifier, instanceSpec, mode)),
          CollectTriggerReqHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_DataCollectCollectTriggerReqId),
          CollectCustomDataReqHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_DataCollectCollectCustomDataReqId){
        ConstructSkeleton(mode);
    }

    explicit HozonInterface_DataCollectSkeleton(const ara::com::InstanceIdentifierContainer instanceContainer,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::interface::datacollect::HozonInterface_DataCollect::ServiceIdentifier, instanceContainer, mode)),
          CollectTriggerReqHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_DataCollectCollectTriggerReqId),
          CollectCustomDataReqHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_DataCollectCollectCustomDataReqId){
        ConstructSkeleton(mode);
    }

    HozonInterface_DataCollectSkeleton(const HozonInterface_DataCollectSkeleton&) = delete;

    HozonInterface_DataCollectSkeleton(HozonInterface_DataCollectSkeleton&&) = default;
    HozonInterface_DataCollectSkeleton& operator=(HozonInterface_DataCollectSkeleton&&) = default;
    HozonInterface_DataCollectSkeleton(ConstructionToken&& token) noexcept 
        : skeletonAdapter(token.GetSkeletonAdapter()),
          CollectTriggerReqHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_DataCollectCollectTriggerReqId),
          CollectCustomDataReqHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_DataCollectCollectCustomDataReqId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifier instanceId,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::interface::datacollect::HozonInterface_DataCollect::ServiceIdentifier, instanceId, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::core::InstanceSpecifier instanceSpec,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::interface::datacollect::HozonInterface_DataCollect::ServiceIdentifier, instanceSpec, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifierContainer instanceIdContainer,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::interface::datacollect::HozonInterface_DataCollect::ServiceIdentifier, instanceIdContainer, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> PreConstructResult(
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter>& preSkeletonAdapter, const ara::com::MethodCallProcessingMode mode)
    {
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            if (mode == ara::com::MethodCallProcessingMode::kEvent) {
                if(!preSkeletonAdapter->SetMethodThreadNumber(preSkeletonAdapter->GetMethodThreadNumber(2U), 1024U)) {
                    result = false;
                    ara::core::ErrorCode errorcode(ara::com::ComErrc::kNetworkBindingFailure);
                    initResult.EmplaceError(errorcode);
                    break;
                }
            }
            initResult = preSkeletonAdapter->InitializeMethod<CollectTriggerReqOutput>(methods::HozonInterface_DataCollectCollectTriggerReqId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<CollectCustomDataReqOutput>(methods::HozonInterface_DataCollectCollectCustomDataReqId);
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

    virtual ~HozonInterface_DataCollectSkeleton()
    {
        StopOfferService();
    }

    void OfferService()
    {
        skeletonAdapter->RegisterE2EErrorHandler(&HozonInterface_DataCollectSkeleton::E2EErrorHandler, *this);
        skeletonAdapter->RegisterMethod(&HozonInterface_DataCollectSkeleton::CollectTriggerReq, *this, methods::HozonInterface_DataCollectCollectTriggerReqId);
        skeletonAdapter->RegisterMethod(&HozonInterface_DataCollectSkeleton::CollectCustomDataReq, *this, methods::HozonInterface_DataCollectCollectCustomDataReqId);
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
    virtual CollectTriggerReqOutput CollectTriggerReq(const ::hozon::datacollect::CollectTrigger& collectTrigger) = 0;
    virtual CollectCustomDataReqOutput CollectCustomDataReq(const ::hozon::datacollect::CustomCollectData& customData) = 0;
    virtual void E2EErrorHandler(ara::com::e2e::E2EErrorCode, ara::com::e2e::DataID, ara::com::e2e::MessageCounter){}

    methods::CollectTriggerReqHandle CollectTriggerReqHandle;
    methods::CollectCustomDataReqHandle CollectCustomDataReqHandle;
};
} // namespace skeleton
} // namespace datacollect
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_DATACOLLECT_HOZONINTERFACE_DATACOLLECT_SKELETON_H
