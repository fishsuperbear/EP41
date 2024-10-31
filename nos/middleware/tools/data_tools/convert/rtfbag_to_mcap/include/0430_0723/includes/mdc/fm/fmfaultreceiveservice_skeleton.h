/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_FM_FMFAULTRECEIVESERVICE_SKELETON_H
#define MDC_FM_FMFAULTRECEIVESERVICE_SKELETON_H

#include "ara/com/internal/skeleton/skeleton_adapter.h"
#include "ara/com/internal/skeleton/event_adapter.h"
#include "ara/com/internal/skeleton/field_adapter.h"
#include "ara/com/internal/skeleton/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "mdc/fm/fmfaultreceiveservice_common.h"
#include <cstdint>

namespace mdc {
namespace fm {
namespace skeleton {
namespace events
{
    using FaultEvent = ara::com::internal::skeleton::event::EventAdapter<::hozon::fm::HzFaultData>;
    using NotifyFaultStateError = ara::com::internal::skeleton::event::EventAdapter<::hozon::fm::HzFaultAnalysisEvent>;
    using NotifyFaultEventError = ara::com::internal::skeleton::event::EventAdapter<::hozon::fm::HzFaultAnalysisEvent>;
    static constexpr ara::com::internal::EntityId FmFaultReceiveServiceFaultEventId = 19654U; //FaultEvent_event_hash
    static constexpr ara::com::internal::EntityId FmFaultReceiveServiceNotifyFaultStateErrorId = 63915U; //NotifyFaultStateError_event_hash
    static constexpr ara::com::internal::EntityId FmFaultReceiveServiceNotifyFaultEventErrorId = 20316U; //NotifyFaultEventError_event_hash
}

namespace methods
{
    using AlarmReportHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using AlarmReport_AsyncHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using GetDataCollectionFileHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using RegistIntInterestFaultHandle = ara::com::internal::skeleton::method::MethodAdapter;
    static constexpr ara::com::internal::EntityId FmFaultReceiveServiceAlarmReportId = 30975U; //AlarmReport_method_hash
    static constexpr ara::com::internal::EntityId FmFaultReceiveServiceAlarmReport_AsyncId = 51475U; //AlarmReport_Async_method_hash
    static constexpr ara::com::internal::EntityId FmFaultReceiveServiceGetDataCollectionFileId = 17955U; //GetDataCollectionFile_method_hash
    static constexpr ara::com::internal::EntityId FmFaultReceiveServiceRegistIntInterestFaultId = 32336U; //RegistIntInterestFault_method_hash
}

namespace fields
{
}

class FmFaultReceiveServiceSkeleton {
private:
    std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> skeletonAdapter;
    void ConstructSkeleton(const ara::com::MethodCallProcessingMode mode)
    {
        if (mode == ara::com::MethodCallProcessingMode::kEvent) {
            if (!(skeletonAdapter->SetMethodThreadNumber(skeletonAdapter->GetMethodThreadNumber(4U), 1024U))) {
#ifndef NOT_SUPPORT_EXCEPTIONS
                ara::core::ErrorCode errorcode(ara::com::ComErrc::kNetworkBindingFailure);
                throw ara::com::ComException(std::move(errorcode));
#else
                std::cerr << "Error: Not support exception, create skeleton failed!"<< std::endl;
#endif
            }
        }
        const ara::core::Result<void> resultFaultEvent = skeletonAdapter->InitializeEvent(FaultEvent);
        ThrowError(resultFaultEvent);
        const ara::core::Result<void> resultNotifyFaultStateError = skeletonAdapter->InitializeEvent(NotifyFaultStateError);
        ThrowError(resultNotifyFaultStateError);
        const ara::core::Result<void> resultNotifyFaultEventError = skeletonAdapter->InitializeEvent(NotifyFaultEventError);
        ThrowError(resultNotifyFaultEventError);
        const ara::core::Result<void> resultAlarmReport = skeletonAdapter->InitializeMethod<ara::core::Future<AlarmReportOutput>>(methods::FmFaultReceiveServiceAlarmReportId);
        ThrowError(resultAlarmReport);
        const ara::core::Result<void> resultAlarmReport_Async = skeletonAdapter->InitializeMethod<AlarmReport_AsyncOutput>(methods::FmFaultReceiveServiceAlarmReport_AsyncId);
        ThrowError(resultAlarmReport_Async);
        const ara::core::Result<void> resultGetDataCollectionFile = skeletonAdapter->InitializeMethod<ara::core::Future<GetDataCollectionFileOutput>>(methods::FmFaultReceiveServiceGetDataCollectionFileId);
        ThrowError(resultGetDataCollectionFile);
        const ara::core::Result<void> resultRegistIntInterestFault = skeletonAdapter->InitializeMethod<RegistIntInterestFaultOutput>(methods::FmFaultReceiveServiceRegistIntInterestFaultId);
        ThrowError(resultRegistIntInterestFault);
    }

    FmFaultReceiveServiceSkeleton& operator=(const FmFaultReceiveServiceSkeleton&) = delete;

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
    using AlarmReportOutput = mdc::fm::methods::AlarmReport::Output;
    
    using AlarmReport_AsyncOutput = void;
    
    using GetDataCollectionFileOutput = mdc::fm::methods::GetDataCollectionFile::Output;
    
    using RegistIntInterestFaultOutput = void;
    
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
    explicit FmFaultReceiveServiceSkeleton(const ara::com::InstanceIdentifier& instanceId,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        : skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::mdc::fm::FmFaultReceiveService::ServiceIdentifier, instanceId, mode)),
          FaultEvent(skeletonAdapter->GetSkeleton(), events::FmFaultReceiveServiceFaultEventId),
          NotifyFaultStateError(skeletonAdapter->GetSkeleton(), events::FmFaultReceiveServiceNotifyFaultStateErrorId),
          NotifyFaultEventError(skeletonAdapter->GetSkeleton(), events::FmFaultReceiveServiceNotifyFaultEventErrorId),
          AlarmReportHandle(skeletonAdapter->GetSkeleton(), methods::FmFaultReceiveServiceAlarmReportId),
          AlarmReport_AsyncHandle(skeletonAdapter->GetSkeleton(), methods::FmFaultReceiveServiceAlarmReport_AsyncId),
          GetDataCollectionFileHandle(skeletonAdapter->GetSkeleton(), methods::FmFaultReceiveServiceGetDataCollectionFileId),
          RegistIntInterestFaultHandle(skeletonAdapter->GetSkeleton(), methods::FmFaultReceiveServiceRegistIntInterestFaultId){
        ConstructSkeleton(mode);
    }

    explicit FmFaultReceiveServiceSkeleton(const ara::core::InstanceSpecifier& instanceSpec,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::mdc::fm::FmFaultReceiveService::ServiceIdentifier, instanceSpec, mode)),
          FaultEvent(skeletonAdapter->GetSkeleton(), events::FmFaultReceiveServiceFaultEventId),
          NotifyFaultStateError(skeletonAdapter->GetSkeleton(), events::FmFaultReceiveServiceNotifyFaultStateErrorId),
          NotifyFaultEventError(skeletonAdapter->GetSkeleton(), events::FmFaultReceiveServiceNotifyFaultEventErrorId),
          AlarmReportHandle(skeletonAdapter->GetSkeleton(), methods::FmFaultReceiveServiceAlarmReportId),
          AlarmReport_AsyncHandle(skeletonAdapter->GetSkeleton(), methods::FmFaultReceiveServiceAlarmReport_AsyncId),
          GetDataCollectionFileHandle(skeletonAdapter->GetSkeleton(), methods::FmFaultReceiveServiceGetDataCollectionFileId),
          RegistIntInterestFaultHandle(skeletonAdapter->GetSkeleton(), methods::FmFaultReceiveServiceRegistIntInterestFaultId){
        ConstructSkeleton(mode);
    }

    explicit FmFaultReceiveServiceSkeleton(const ara::com::InstanceIdentifierContainer instanceContainer,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::mdc::fm::FmFaultReceiveService::ServiceIdentifier, instanceContainer, mode)),
          FaultEvent(skeletonAdapter->GetSkeleton(), events::FmFaultReceiveServiceFaultEventId),
          NotifyFaultStateError(skeletonAdapter->GetSkeleton(), events::FmFaultReceiveServiceNotifyFaultStateErrorId),
          NotifyFaultEventError(skeletonAdapter->GetSkeleton(), events::FmFaultReceiveServiceNotifyFaultEventErrorId),
          AlarmReportHandle(skeletonAdapter->GetSkeleton(), methods::FmFaultReceiveServiceAlarmReportId),
          AlarmReport_AsyncHandle(skeletonAdapter->GetSkeleton(), methods::FmFaultReceiveServiceAlarmReport_AsyncId),
          GetDataCollectionFileHandle(skeletonAdapter->GetSkeleton(), methods::FmFaultReceiveServiceGetDataCollectionFileId),
          RegistIntInterestFaultHandle(skeletonAdapter->GetSkeleton(), methods::FmFaultReceiveServiceRegistIntInterestFaultId){
        ConstructSkeleton(mode);
    }

    FmFaultReceiveServiceSkeleton(const FmFaultReceiveServiceSkeleton&) = delete;

    FmFaultReceiveServiceSkeleton(FmFaultReceiveServiceSkeleton&&) = default;
    FmFaultReceiveServiceSkeleton& operator=(FmFaultReceiveServiceSkeleton&&) = default;
    FmFaultReceiveServiceSkeleton(ConstructionToken&& token) noexcept 
        : skeletonAdapter(token.GetSkeletonAdapter()),
          FaultEvent(skeletonAdapter->GetSkeleton(), events::FmFaultReceiveServiceFaultEventId),
          NotifyFaultStateError(skeletonAdapter->GetSkeleton(), events::FmFaultReceiveServiceNotifyFaultStateErrorId),
          NotifyFaultEventError(skeletonAdapter->GetSkeleton(), events::FmFaultReceiveServiceNotifyFaultEventErrorId),
          AlarmReportHandle(skeletonAdapter->GetSkeleton(), methods::FmFaultReceiveServiceAlarmReportId),
          AlarmReport_AsyncHandle(skeletonAdapter->GetSkeleton(), methods::FmFaultReceiveServiceAlarmReport_AsyncId),
          GetDataCollectionFileHandle(skeletonAdapter->GetSkeleton(), methods::FmFaultReceiveServiceGetDataCollectionFileId),
          RegistIntInterestFaultHandle(skeletonAdapter->GetSkeleton(), methods::FmFaultReceiveServiceRegistIntInterestFaultId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifier instanceId,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::mdc::fm::FmFaultReceiveService::ServiceIdentifier, instanceId, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::core::InstanceSpecifier instanceSpec,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::mdc::fm::FmFaultReceiveService::ServiceIdentifier, instanceSpec, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifierContainer instanceIdContainer,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::mdc::fm::FmFaultReceiveService::ServiceIdentifier, instanceIdContainer, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> PreConstructResult(
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter>& preSkeletonAdapter, const ara::com::MethodCallProcessingMode mode)
    {
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            if (mode == ara::com::MethodCallProcessingMode::kEvent) {
                if(!preSkeletonAdapter->SetMethodThreadNumber(preSkeletonAdapter->GetMethodThreadNumber(4U), 1024U)) {
                    result = false;
                    ara::core::ErrorCode errorcode(ara::com::ComErrc::kNetworkBindingFailure);
                    initResult.EmplaceError(errorcode);
                    break;
                }
            }
            const events::FaultEvent FaultEvent(preSkeletonAdapter->GetSkeleton(), events::FmFaultReceiveServiceFaultEventId);
            initResult = preSkeletonAdapter->InitializeEvent(FaultEvent);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const events::NotifyFaultStateError NotifyFaultStateError(preSkeletonAdapter->GetSkeleton(), events::FmFaultReceiveServiceNotifyFaultStateErrorId);
            initResult = preSkeletonAdapter->InitializeEvent(NotifyFaultStateError);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const events::NotifyFaultEventError NotifyFaultEventError(preSkeletonAdapter->GetSkeleton(), events::FmFaultReceiveServiceNotifyFaultEventErrorId);
            initResult = preSkeletonAdapter->InitializeEvent(NotifyFaultEventError);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<AlarmReportOutput>>(methods::FmFaultReceiveServiceAlarmReportId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<AlarmReport_AsyncOutput>(methods::FmFaultReceiveServiceAlarmReport_AsyncId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<GetDataCollectionFileOutput>>(methods::FmFaultReceiveServiceGetDataCollectionFileId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<RegistIntInterestFaultOutput>(methods::FmFaultReceiveServiceRegistIntInterestFaultId);
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

    virtual ~FmFaultReceiveServiceSkeleton()
    {
        StopOfferService();
    }

    void OfferService()
    {
        skeletonAdapter->RegisterE2EErrorHandler(&FmFaultReceiveServiceSkeleton::E2EErrorHandler, *this);
        skeletonAdapter->RegisterMethod(&FmFaultReceiveServiceSkeleton::AlarmReport, *this, methods::FmFaultReceiveServiceAlarmReportId);
        skeletonAdapter->RegisterMethod(&FmFaultReceiveServiceSkeleton::AlarmReport_Async, *this, methods::FmFaultReceiveServiceAlarmReport_AsyncId);
        skeletonAdapter->RegisterMethod(&FmFaultReceiveServiceSkeleton::GetDataCollectionFile, *this, methods::FmFaultReceiveServiceGetDataCollectionFileId);
        skeletonAdapter->RegisterMethod(&FmFaultReceiveServiceSkeleton::RegistIntInterestFault, *this, methods::FmFaultReceiveServiceRegistIntInterestFaultId);
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
    virtual ara::core::Future<AlarmReportOutput> AlarmReport(const ::hozon::fm::HzFaultData& faultMsg) = 0;
    virtual AlarmReport_AsyncOutput AlarmReport_Async(const ::hozon::fm::HzFaultData& faultMsg) = 0;
    virtual ara::core::Future<GetDataCollectionFileOutput> GetDataCollectionFile() = 0;
    virtual RegistIntInterestFaultOutput RegistIntInterestFault(const ::hozon::fm::HzFaultItemVector& faultItemVector, const ::hozon::fm::HzFaultClusterVector& faultClusterVector) = 0;
    virtual void E2EErrorHandler(ara::com::e2e::E2EErrorCode, ara::com::e2e::DataID, ara::com::e2e::MessageCounter){}

    events::FaultEvent FaultEvent;
    events::NotifyFaultStateError NotifyFaultStateError;
    events::NotifyFaultEventError NotifyFaultEventError;
    methods::AlarmReportHandle AlarmReportHandle;
    methods::AlarmReport_AsyncHandle AlarmReport_AsyncHandle;
    methods::GetDataCollectionFileHandle GetDataCollectionFileHandle;
    methods::RegistIntInterestFaultHandle RegistIntInterestFaultHandle;
};
} // namespace skeleton
} // namespace fm
} // namespace mdc

#endif // MDC_FM_FMFAULTRECEIVESERVICE_SKELETON_H
