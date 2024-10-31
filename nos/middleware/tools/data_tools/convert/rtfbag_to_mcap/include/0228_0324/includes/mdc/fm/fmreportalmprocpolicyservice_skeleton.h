/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_FM_FMREPORTALMPROCPOLICYSERVICE_SKELETON_H
#define MDC_FM_FMREPORTALMPROCPOLICYSERVICE_SKELETON_H

#include "ara/com/internal/skeleton/skeleton_adapter.h"
#include "ara/com/internal/skeleton/event_adapter.h"
#include "ara/com/internal/skeleton/field_adapter.h"
#include "ara/com/internal/skeleton/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "mdc/fm/fmreportalmprocpolicyservice_common.h"
#include <cstdint>

namespace mdc {
namespace fm {
namespace skeleton {
namespace events
{
}

namespace methods
{
    using ReportAlmProcPolicyHandle = ara::com::internal::skeleton::method::MethodAdapter;
    static constexpr ara::com::internal::EntityId FmReportAlmProcPolicyServiceReportAlmProcPolicyId = 6033U; //ReportAlmProcPolicy_method_hash
}

namespace fields
{
}

class FmReportAlmProcPolicyServiceSkeleton {
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
        const ara::core::Result<void> resultReportAlmProcPolicy = skeletonAdapter->InitializeMethod<ara::core::Future<ReportAlmProcPolicyOutput>>(methods::FmReportAlmProcPolicyServiceReportAlmProcPolicyId);
        ThrowError(resultReportAlmProcPolicy);
    }

    FmReportAlmProcPolicyServiceSkeleton& operator=(const FmReportAlmProcPolicyServiceSkeleton&) = delete;

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
    using ReportAlmProcPolicyOutput = mdc::fm::methods::ReportAlmProcPolicy::Output;
    
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
    explicit FmReportAlmProcPolicyServiceSkeleton(const ara::com::InstanceIdentifier& instanceId,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        : skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::mdc::fm::FmReportAlmProcPolicyService::ServiceIdentifier, instanceId, mode)),
          ReportAlmProcPolicyHandle(skeletonAdapter->GetSkeleton(), methods::FmReportAlmProcPolicyServiceReportAlmProcPolicyId){
        ConstructSkeleton(mode);
    }

    explicit FmReportAlmProcPolicyServiceSkeleton(const ara::core::InstanceSpecifier& instanceSpec,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::mdc::fm::FmReportAlmProcPolicyService::ServiceIdentifier, instanceSpec, mode)),
          ReportAlmProcPolicyHandle(skeletonAdapter->GetSkeleton(), methods::FmReportAlmProcPolicyServiceReportAlmProcPolicyId){
        ConstructSkeleton(mode);
    }

    explicit FmReportAlmProcPolicyServiceSkeleton(const ara::com::InstanceIdentifierContainer instanceContainer,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::mdc::fm::FmReportAlmProcPolicyService::ServiceIdentifier, instanceContainer, mode)),
          ReportAlmProcPolicyHandle(skeletonAdapter->GetSkeleton(), methods::FmReportAlmProcPolicyServiceReportAlmProcPolicyId){
        ConstructSkeleton(mode);
    }

    FmReportAlmProcPolicyServiceSkeleton(const FmReportAlmProcPolicyServiceSkeleton&) = delete;

    FmReportAlmProcPolicyServiceSkeleton(FmReportAlmProcPolicyServiceSkeleton&&) = default;
    FmReportAlmProcPolicyServiceSkeleton& operator=(FmReportAlmProcPolicyServiceSkeleton&&) = default;
    FmReportAlmProcPolicyServiceSkeleton(ConstructionToken&& token) noexcept 
        : skeletonAdapter(token.GetSkeletonAdapter()),
          ReportAlmProcPolicyHandle(skeletonAdapter->GetSkeleton(), methods::FmReportAlmProcPolicyServiceReportAlmProcPolicyId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifier instanceId,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::mdc::fm::FmReportAlmProcPolicyService::ServiceIdentifier, instanceId, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::core::InstanceSpecifier instanceSpec,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::mdc::fm::FmReportAlmProcPolicyService::ServiceIdentifier, instanceSpec, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifierContainer instanceIdContainer,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::mdc::fm::FmReportAlmProcPolicyService::ServiceIdentifier, instanceIdContainer, mode);
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
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<ReportAlmProcPolicyOutput>>(methods::FmReportAlmProcPolicyServiceReportAlmProcPolicyId);
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

    virtual ~FmReportAlmProcPolicyServiceSkeleton()
    {
        StopOfferService();
    }

    void OfferService()
    {
        skeletonAdapter->RegisterE2EErrorHandler(&FmReportAlmProcPolicyServiceSkeleton::E2EErrorHandler, *this);
        skeletonAdapter->RegisterMethod(&FmReportAlmProcPolicyServiceSkeleton::ReportAlmProcPolicy, *this, methods::FmReportAlmProcPolicyServiceReportAlmProcPolicyId);
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
    virtual ara::core::Future<ReportAlmProcPolicyOutput> ReportAlmProcPolicy(const ::UInt8& retryTimes, const ::UInt8& procPolicy) = 0;
    virtual void E2EErrorHandler(ara::com::e2e::E2EErrorCode, ara::com::e2e::DataID, ara::com::e2e::MessageCounter){}

    methods::ReportAlmProcPolicyHandle ReportAlmProcPolicyHandle;
};
} // namespace skeleton
} // namespace fm
} // namespace mdc

#endif // MDC_FM_FMREPORTALMPROCPOLICYSERVICE_SKELETON_H
