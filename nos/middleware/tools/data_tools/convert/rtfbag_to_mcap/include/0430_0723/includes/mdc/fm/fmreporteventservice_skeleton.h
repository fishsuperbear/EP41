/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_FM_FMREPORTEVENTSERVICE_SKELETON_H
#define MDC_FM_FMREPORTEVENTSERVICE_SKELETON_H

#include "ara/com/internal/skeleton/skeleton_adapter.h"
#include "ara/com/internal/skeleton/event_adapter.h"
#include "ara/com/internal/skeleton/field_adapter.h"
#include "ara/com/internal/skeleton/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "mdc/fm/fmreporteventservice_common.h"
#include <cstdint>

namespace mdc {
namespace fm {
namespace skeleton {
namespace events
{
}

namespace methods
{
    using ReportFaultHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using ReportCheckPointHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using ReportProcStateHandle = ara::com::internal::skeleton::method::MethodAdapter;
    static constexpr ara::com::internal::EntityId FmReportEventServiceReportFaultId = 14388U; //ReportFault_method_hash
    static constexpr ara::com::internal::EntityId FmReportEventServiceReportCheckPointId = 3322U; //ReportCheckPoint_method_hash
    static constexpr ara::com::internal::EntityId FmReportEventServiceReportProcStateId = 24315U; //ReportProcState_method_hash
}

namespace fields
{
}

class FmReportEventServiceSkeleton {
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
        const ara::core::Result<void> resultReportFault = skeletonAdapter->InitializeMethod<ara::core::Future<ReportFaultOutput>>(methods::FmReportEventServiceReportFaultId);
        ThrowError(resultReportFault);
        const ara::core::Result<void> resultReportCheckPoint = skeletonAdapter->InitializeMethod<ara::core::Future<ReportCheckPointOutput>>(methods::FmReportEventServiceReportCheckPointId);
        ThrowError(resultReportCheckPoint);
        const ara::core::Result<void> resultReportProcState = skeletonAdapter->InitializeMethod<ara::core::Future<ReportProcStateOutput>>(methods::FmReportEventServiceReportProcStateId);
        ThrowError(resultReportProcState);
    }

    FmReportEventServiceSkeleton& operator=(const FmReportEventServiceSkeleton&) = delete;

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
    using ReportFaultOutput = mdc::fm::methods::ReportFault::Output;
    
    using ReportCheckPointOutput = mdc::fm::methods::ReportCheckPoint::Output;
    
    using ReportProcStateOutput = mdc::fm::methods::ReportProcState::Output;
    
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
    explicit FmReportEventServiceSkeleton(const ara::com::InstanceIdentifier& instanceId,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        : skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::mdc::fm::FmReportEventService::ServiceIdentifier, instanceId, mode)),
          ReportFaultHandle(skeletonAdapter->GetSkeleton(), methods::FmReportEventServiceReportFaultId),
          ReportCheckPointHandle(skeletonAdapter->GetSkeleton(), methods::FmReportEventServiceReportCheckPointId),
          ReportProcStateHandle(skeletonAdapter->GetSkeleton(), methods::FmReportEventServiceReportProcStateId){
        ConstructSkeleton(mode);
    }

    explicit FmReportEventServiceSkeleton(const ara::core::InstanceSpecifier& instanceSpec,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::mdc::fm::FmReportEventService::ServiceIdentifier, instanceSpec, mode)),
          ReportFaultHandle(skeletonAdapter->GetSkeleton(), methods::FmReportEventServiceReportFaultId),
          ReportCheckPointHandle(skeletonAdapter->GetSkeleton(), methods::FmReportEventServiceReportCheckPointId),
          ReportProcStateHandle(skeletonAdapter->GetSkeleton(), methods::FmReportEventServiceReportProcStateId){
        ConstructSkeleton(mode);
    }

    explicit FmReportEventServiceSkeleton(const ara::com::InstanceIdentifierContainer instanceContainer,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::mdc::fm::FmReportEventService::ServiceIdentifier, instanceContainer, mode)),
          ReportFaultHandle(skeletonAdapter->GetSkeleton(), methods::FmReportEventServiceReportFaultId),
          ReportCheckPointHandle(skeletonAdapter->GetSkeleton(), methods::FmReportEventServiceReportCheckPointId),
          ReportProcStateHandle(skeletonAdapter->GetSkeleton(), methods::FmReportEventServiceReportProcStateId){
        ConstructSkeleton(mode);
    }

    FmReportEventServiceSkeleton(const FmReportEventServiceSkeleton&) = delete;

    FmReportEventServiceSkeleton(FmReportEventServiceSkeleton&&) = default;
    FmReportEventServiceSkeleton& operator=(FmReportEventServiceSkeleton&&) = default;
    FmReportEventServiceSkeleton(ConstructionToken&& token) noexcept 
        : skeletonAdapter(token.GetSkeletonAdapter()),
          ReportFaultHandle(skeletonAdapter->GetSkeleton(), methods::FmReportEventServiceReportFaultId),
          ReportCheckPointHandle(skeletonAdapter->GetSkeleton(), methods::FmReportEventServiceReportCheckPointId),
          ReportProcStateHandle(skeletonAdapter->GetSkeleton(), methods::FmReportEventServiceReportProcStateId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifier instanceId,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::mdc::fm::FmReportEventService::ServiceIdentifier, instanceId, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::core::InstanceSpecifier instanceSpec,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::mdc::fm::FmReportEventService::ServiceIdentifier, instanceSpec, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifierContainer instanceIdContainer,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::mdc::fm::FmReportEventService::ServiceIdentifier, instanceIdContainer, mode);
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
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<ReportFaultOutput>>(methods::FmReportEventServiceReportFaultId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<ReportCheckPointOutput>>(methods::FmReportEventServiceReportCheckPointId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<ReportProcStateOutput>>(methods::FmReportEventServiceReportProcStateId);
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

    virtual ~FmReportEventServiceSkeleton()
    {
        StopOfferService();
    }

    void OfferService()
    {
        skeletonAdapter->RegisterE2EErrorHandler(&FmReportEventServiceSkeleton::E2EErrorHandler, *this);
        skeletonAdapter->RegisterMethod(&FmReportEventServiceSkeleton::ReportFault, *this, methods::FmReportEventServiceReportFaultId);
        skeletonAdapter->RegisterMethod(&FmReportEventServiceSkeleton::ReportCheckPoint, *this, methods::FmReportEventServiceReportCheckPointId);
        skeletonAdapter->RegisterMethod(&FmReportEventServiceSkeleton::ReportProcState, *this, methods::FmReportEventServiceReportProcStateId);
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
    virtual ara::core::Future<ReportFaultOutput> ReportFault(const ::mdc::fm::FmFaultData& faultData) = 0;
    virtual ara::core::Future<ReportCheckPointOutput> ReportCheckPoint(const ::String& procName) = 0;
    virtual ara::core::Future<ReportProcStateOutput> ReportProcState(const ::String& procName, const ::uint8_t& state) = 0;
    virtual void E2EErrorHandler(ara::com::e2e::E2EErrorCode, ara::com::e2e::DataID, ara::com::e2e::MessageCounter){}

    methods::ReportFaultHandle ReportFaultHandle;
    methods::ReportCheckPointHandle ReportCheckPointHandle;
    methods::ReportProcStateHandle ReportProcStateHandle;
};
} // namespace skeleton
} // namespace fm
} // namespace mdc

#endif // MDC_FM_FMREPORTEVENTSERVICE_SKELETON_H
