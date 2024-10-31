/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_FM_FMQUERYSERVICE_SKELETON_H
#define MDC_FM_FMQUERYSERVICE_SKELETON_H

#include "ara/com/internal/skeleton/skeleton_adapter.h"
#include "ara/com/internal/skeleton/event_adapter.h"
#include "ara/com/internal/skeleton/field_adapter.h"
#include "ara/com/internal/skeleton/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "mdc/fm/fmqueryservice_common.h"
#include <cstdint>

namespace mdc {
namespace fm {
namespace skeleton {
namespace events
{
}

namespace methods
{
    using QueryFaultDetailHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using QueryFaultOnFlagHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using QueryFaultStatisticHandle = ara::com::internal::skeleton::method::MethodAdapter;
    static constexpr ara::com::internal::EntityId FmQueryServiceQueryFaultDetailId = 36459U; //QueryFaultDetail_method_hash
    static constexpr ara::com::internal::EntityId FmQueryServiceQueryFaultOnFlagId = 43267U; //QueryFaultOnFlag_method_hash
    static constexpr ara::com::internal::EntityId FmQueryServiceQueryFaultStatisticId = 26435U; //QueryFaultStatistic_method_hash
}

namespace fields
{
}

class FmQueryServiceSkeleton {
private:
    std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> skeletonAdapter;
    void ConstructSkeleton(const ara::com::MethodCallProcessingMode mode)
    {
        if (mode == ara::com::MethodCallProcessingMode::kEvent) {
            if (!(skeletonAdapter->SetMethodThreadNumber(skeletonAdapter->GetMethodThreadNumber(3U), 1024U))) {
#ifndef NOT_SUPPORT_EXCEPTIONS
                ara::core::ErrorCode errorcode(ara::com::ComErrc::kNetworkBindingFailure);
                throw ara::com::ComException(std::move(errorcode));
#else
                std::cerr << "Error: Not support exception, create skeleton failed!"<< std::endl;
#endif
            }
        }
        const ara::core::Result<void> resultQueryFaultDetail = skeletonAdapter->InitializeMethod<ara::core::Future<QueryFaultDetailOutput>>(methods::FmQueryServiceQueryFaultDetailId);
        ThrowError(resultQueryFaultDetail);
        const ara::core::Result<void> resultQueryFaultOnFlag = skeletonAdapter->InitializeMethod<ara::core::Future<QueryFaultOnFlagOutput>>(methods::FmQueryServiceQueryFaultOnFlagId);
        ThrowError(resultQueryFaultOnFlag);
        const ara::core::Result<void> resultQueryFaultStatistic = skeletonAdapter->InitializeMethod<ara::core::Future<QueryFaultStatisticOutput>>(methods::FmQueryServiceQueryFaultStatisticId);
        ThrowError(resultQueryFaultStatistic);
    }

    FmQueryServiceSkeleton& operator=(const FmQueryServiceSkeleton&) = delete;

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
    using QueryFaultDetailOutput = mdc::fm::methods::QueryFaultDetail::Output;
    
    using QueryFaultOnFlagOutput = mdc::fm::methods::QueryFaultOnFlag::Output;
    
    using QueryFaultStatisticOutput = mdc::fm::methods::QueryFaultStatistic::Output;
    
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
    explicit FmQueryServiceSkeleton(const ara::com::InstanceIdentifier& instanceId,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        : skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::mdc::fm::FmQueryService::ServiceIdentifier, instanceId, mode)),
          QueryFaultDetailHandle(skeletonAdapter->GetSkeleton(), methods::FmQueryServiceQueryFaultDetailId),
          QueryFaultOnFlagHandle(skeletonAdapter->GetSkeleton(), methods::FmQueryServiceQueryFaultOnFlagId),
          QueryFaultStatisticHandle(skeletonAdapter->GetSkeleton(), methods::FmQueryServiceQueryFaultStatisticId){
        ConstructSkeleton(mode);
    }

    explicit FmQueryServiceSkeleton(const ara::core::InstanceSpecifier& instanceSpec,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::mdc::fm::FmQueryService::ServiceIdentifier, instanceSpec, mode)),
          QueryFaultDetailHandle(skeletonAdapter->GetSkeleton(), methods::FmQueryServiceQueryFaultDetailId),
          QueryFaultOnFlagHandle(skeletonAdapter->GetSkeleton(), methods::FmQueryServiceQueryFaultOnFlagId),
          QueryFaultStatisticHandle(skeletonAdapter->GetSkeleton(), methods::FmQueryServiceQueryFaultStatisticId){
        ConstructSkeleton(mode);
    }

    explicit FmQueryServiceSkeleton(const ara::com::InstanceIdentifierContainer instanceContainer,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::mdc::fm::FmQueryService::ServiceIdentifier, instanceContainer, mode)),
          QueryFaultDetailHandle(skeletonAdapter->GetSkeleton(), methods::FmQueryServiceQueryFaultDetailId),
          QueryFaultOnFlagHandle(skeletonAdapter->GetSkeleton(), methods::FmQueryServiceQueryFaultOnFlagId),
          QueryFaultStatisticHandle(skeletonAdapter->GetSkeleton(), methods::FmQueryServiceQueryFaultStatisticId){
        ConstructSkeleton(mode);
    }

    FmQueryServiceSkeleton(const FmQueryServiceSkeleton&) = delete;

    FmQueryServiceSkeleton(FmQueryServiceSkeleton&&) = default;
    FmQueryServiceSkeleton& operator=(FmQueryServiceSkeleton&&) = default;
    FmQueryServiceSkeleton(ConstructionToken&& token) noexcept 
        : skeletonAdapter(token.GetSkeletonAdapter()),
          QueryFaultDetailHandle(skeletonAdapter->GetSkeleton(), methods::FmQueryServiceQueryFaultDetailId),
          QueryFaultOnFlagHandle(skeletonAdapter->GetSkeleton(), methods::FmQueryServiceQueryFaultOnFlagId),
          QueryFaultStatisticHandle(skeletonAdapter->GetSkeleton(), methods::FmQueryServiceQueryFaultStatisticId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifier instanceId,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::mdc::fm::FmQueryService::ServiceIdentifier, instanceId, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::core::InstanceSpecifier instanceSpec,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::mdc::fm::FmQueryService::ServiceIdentifier, instanceSpec, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifierContainer instanceIdContainer,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::mdc::fm::FmQueryService::ServiceIdentifier, instanceIdContainer, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> PreConstructResult(
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter>& preSkeletonAdapter, const ara::com::MethodCallProcessingMode mode)
    {
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            if (mode == ara::com::MethodCallProcessingMode::kEvent) {
                if(!preSkeletonAdapter->SetMethodThreadNumber(preSkeletonAdapter->GetMethodThreadNumber(3U), 1024U)) {
                    result = false;
                    ara::core::ErrorCode errorcode(ara::com::ComErrc::kNetworkBindingFailure);
                    initResult.EmplaceError(errorcode);
                    break;
                }
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<QueryFaultDetailOutput>>(methods::FmQueryServiceQueryFaultDetailId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<QueryFaultOnFlagOutput>>(methods::FmQueryServiceQueryFaultOnFlagId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<QueryFaultStatisticOutput>>(methods::FmQueryServiceQueryFaultStatisticId);
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

    virtual ~FmQueryServiceSkeleton()
    {
        StopOfferService();
    }

    void OfferService()
    {
        skeletonAdapter->RegisterE2EErrorHandler(&FmQueryServiceSkeleton::E2EErrorHandler, *this);
        skeletonAdapter->RegisterMethod(&FmQueryServiceSkeleton::QueryFaultDetail, *this, methods::FmQueryServiceQueryFaultDetailId);
        skeletonAdapter->RegisterMethod(&FmQueryServiceSkeleton::QueryFaultOnFlag, *this, methods::FmQueryServiceQueryFaultOnFlagId);
        skeletonAdapter->RegisterMethod(&FmQueryServiceSkeleton::QueryFaultStatistic, *this, methods::FmQueryServiceQueryFaultStatisticId);
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
    virtual ara::core::Future<QueryFaultDetailOutput> QueryFaultDetail() = 0;
    virtual ara::core::Future<QueryFaultOnFlagOutput> QueryFaultOnFlag(const ::UInt32& flag) = 0;
    virtual ara::core::Future<QueryFaultStatisticOutput> QueryFaultStatistic(const ::UInt32& flag) = 0;
    virtual void E2EErrorHandler(ara::com::e2e::E2EErrorCode, ara::com::e2e::DataID, ara::com::e2e::MessageCounter){}

    methods::QueryFaultDetailHandle QueryFaultDetailHandle;
    methods::QueryFaultOnFlagHandle QueryFaultOnFlagHandle;
    methods::QueryFaultStatisticHandle QueryFaultStatisticHandle;
};
} // namespace skeleton
} // namespace fm
} // namespace mdc

#endif // MDC_FM_FMQUERYSERVICE_SKELETON_H
