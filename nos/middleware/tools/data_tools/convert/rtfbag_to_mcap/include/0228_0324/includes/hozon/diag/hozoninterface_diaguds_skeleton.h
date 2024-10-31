/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_DIAG_HOZONINTERFACE_DIAGUDS_SKELETON_H
#define HOZON_DIAG_HOZONINTERFACE_DIAGUDS_SKELETON_H

#include "ara/com/internal/skeleton/skeleton_adapter.h"
#include "ara/com/internal/skeleton/event_adapter.h"
#include "ara/com/internal/skeleton/field_adapter.h"
#include "ara/com/internal/skeleton/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/diag/hozoninterface_diaguds_common.h"
#include <cstdint>

namespace hozon {
namespace diag {
namespace skeleton {
namespace events
{
    using SocUdsReq = ara::com::internal::skeleton::event::EventAdapter<::hozon::diag::UdsFrame>;
    static constexpr ara::com::internal::EntityId HozonInterface_DiagUdsSocUdsReqId = 2396U; //SocUdsReq_event_hash
}

namespace methods
{
    using McuUdsResHandle = ara::com::internal::skeleton::method::MethodAdapter;
    static constexpr ara::com::internal::EntityId HozonInterface_DiagUdsMcuUdsResId = 28543U; //McuUdsRes_method_hash
}

namespace fields
{
}

class HozonInterface_DiagUdsSkeleton {
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
        const ara::core::Result<void> resultSocUdsReq = skeletonAdapter->InitializeEvent(SocUdsReq);
        ThrowError(resultSocUdsReq);
        const ara::core::Result<void> resultMcuUdsRes = skeletonAdapter->InitializeMethod<ara::core::Future<McuUdsResOutput>>(methods::HozonInterface_DiagUdsMcuUdsResId);
        ThrowError(resultMcuUdsRes);
    }

    HozonInterface_DiagUdsSkeleton& operator=(const HozonInterface_DiagUdsSkeleton&) = delete;

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
    using McuUdsResOutput = hozon::diag::methods::McuUdsRes::Output;
    
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
    explicit HozonInterface_DiagUdsSkeleton(const ara::com::InstanceIdentifier& instanceId,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        : skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::diag::HozonInterface_DiagUds::ServiceIdentifier, instanceId, mode)),
          SocUdsReq(skeletonAdapter->GetSkeleton(), events::HozonInterface_DiagUdsSocUdsReqId),
          McuUdsResHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_DiagUdsMcuUdsResId){
        ConstructSkeleton(mode);
    }

    explicit HozonInterface_DiagUdsSkeleton(const ara::core::InstanceSpecifier& instanceSpec,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::diag::HozonInterface_DiagUds::ServiceIdentifier, instanceSpec, mode)),
          SocUdsReq(skeletonAdapter->GetSkeleton(), events::HozonInterface_DiagUdsSocUdsReqId),
          McuUdsResHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_DiagUdsMcuUdsResId){
        ConstructSkeleton(mode);
    }

    explicit HozonInterface_DiagUdsSkeleton(const ara::com::InstanceIdentifierContainer instanceContainer,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::diag::HozonInterface_DiagUds::ServiceIdentifier, instanceContainer, mode)),
          SocUdsReq(skeletonAdapter->GetSkeleton(), events::HozonInterface_DiagUdsSocUdsReqId),
          McuUdsResHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_DiagUdsMcuUdsResId){
        ConstructSkeleton(mode);
    }

    HozonInterface_DiagUdsSkeleton(const HozonInterface_DiagUdsSkeleton&) = delete;

    HozonInterface_DiagUdsSkeleton(HozonInterface_DiagUdsSkeleton&&) = default;
    HozonInterface_DiagUdsSkeleton& operator=(HozonInterface_DiagUdsSkeleton&&) = default;
    HozonInterface_DiagUdsSkeleton(ConstructionToken&& token) noexcept 
        : skeletonAdapter(token.GetSkeletonAdapter()),
          SocUdsReq(skeletonAdapter->GetSkeleton(), events::HozonInterface_DiagUdsSocUdsReqId),
          McuUdsResHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_DiagUdsMcuUdsResId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifier instanceId,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::diag::HozonInterface_DiagUds::ServiceIdentifier, instanceId, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::core::InstanceSpecifier instanceSpec,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::diag::HozonInterface_DiagUds::ServiceIdentifier, instanceSpec, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifierContainer instanceIdContainer,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::diag::HozonInterface_DiagUds::ServiceIdentifier, instanceIdContainer, mode);
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
            const events::SocUdsReq SocUdsReq(preSkeletonAdapter->GetSkeleton(), events::HozonInterface_DiagUdsSocUdsReqId);
            initResult = preSkeletonAdapter->InitializeEvent(SocUdsReq);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<McuUdsResOutput>>(methods::HozonInterface_DiagUdsMcuUdsResId);
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

    virtual ~HozonInterface_DiagUdsSkeleton()
    {
        StopOfferService();
    }

    void OfferService()
    {
        skeletonAdapter->RegisterE2EErrorHandler(&HozonInterface_DiagUdsSkeleton::E2EErrorHandler, *this);
        skeletonAdapter->RegisterMethod(&HozonInterface_DiagUdsSkeleton::McuUdsRes, *this, methods::HozonInterface_DiagUdsMcuUdsResId);
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
    virtual ara::core::Future<McuUdsResOutput> McuUdsRes(const ::hozon::diag::McuUdsResFrame& udsData) = 0;
    virtual void E2EErrorHandler(ara::com::e2e::E2EErrorCode, ara::com::e2e::DataID, ara::com::e2e::MessageCounter){}

    events::SocUdsReq SocUdsReq;
    methods::McuUdsResHandle McuUdsResHandle;
};
} // namespace skeleton
} // namespace diag
} // namespace hozon

#endif // HOZON_DIAG_HOZONINTERFACE_DIAGUDS_SKELETON_H
