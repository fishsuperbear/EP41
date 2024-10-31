/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HM_HZHMSERVICEINTERFACE_SKELETON_H
#define HOZON_HM_HZHMSERVICEINTERFACE_SKELETON_H

#include "ara/com/internal/skeleton/skeleton_adapter.h"
#include "ara/com/internal/skeleton/event_adapter.h"
#include "ara/com/internal/skeleton/field_adapter.h"
#include "ara/com/internal/skeleton/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/hm/hzhmserviceinterface_common.h"
#include <cstdint>

namespace hozon {
namespace hm {
namespace skeleton {
namespace events
{
}

namespace methods
{
    using RegistAliveTaskHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using ReportAliveStatusHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using UnRegistAliveTaskHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using RegistDeadlineTaskHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using ReportDeadlineStatusHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using UnRegistDeadlineTaskHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using RegistLogicTaskHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using ReportLogicCheckpointHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using UnRegistLogicTaskHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using ReportProcAliveCheckpointHandle = ara::com::internal::skeleton::method::MethodAdapter;
    static constexpr ara::com::internal::EntityId HzHmServiceInterfaceRegistAliveTaskId = 3029U; //RegistAliveTask_method_hash
    static constexpr ara::com::internal::EntityId HzHmServiceInterfaceReportAliveStatusId = 25182U; //ReportAliveStatus_method_hash
    static constexpr ara::com::internal::EntityId HzHmServiceInterfaceUnRegistAliveTaskId = 6961U; //UnRegistAliveTask_method_hash
    static constexpr ara::com::internal::EntityId HzHmServiceInterfaceRegistDeadlineTaskId = 62516U; //RegistDeadlineTask_method_hash
    static constexpr ara::com::internal::EntityId HzHmServiceInterfaceReportDeadlineStatusId = 37137U; //ReportDeadlineStatus_method_hash
    static constexpr ara::com::internal::EntityId HzHmServiceInterfaceUnRegistDeadlineTaskId = 37253U; //UnRegistDeadlineTask_method_hash
    static constexpr ara::com::internal::EntityId HzHmServiceInterfaceRegistLogicTaskId = 28627U; //RegistLogicTask_method_hash
    static constexpr ara::com::internal::EntityId HzHmServiceInterfaceReportLogicCheckpointId = 54299U; //ReportLogicCheckpoint_method_hash
    static constexpr ara::com::internal::EntityId HzHmServiceInterfaceUnRegistLogicTaskId = 63805U; //UnRegistLogicTask_method_hash
    static constexpr ara::com::internal::EntityId HzHmServiceInterfaceReportProcAliveCheckpointId = 54530U; //ReportProcAliveCheckpoint_method_hash
}

namespace fields
{
}

class HzHmServiceInterfaceSkeleton {
private:
    std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> skeletonAdapter;
    void ConstructSkeleton(const ara::com::MethodCallProcessingMode mode)
    {
        if (mode == ara::com::MethodCallProcessingMode::kEvent) {
            if (!(skeletonAdapter->SetMethodThreadNumber(skeletonAdapter->GetMethodThreadNumber(10U), 1024U))) {
#ifndef NOT_SUPPORT_EXCEPTIONS
                ara::core::ErrorCode errorcode(ara::com::ComErrc::kNetworkBindingFailure);
                throw ara::com::ComException(std::move(errorcode));
#else
                std::cerr << "Error: Not support exception, create skeleton failed!"<< std::endl;
#endif
            }
        }
        const ara::core::Result<void> resultRegistAliveTask = skeletonAdapter->InitializeMethod<ara::core::Future<RegistAliveTaskOutput>>(methods::HzHmServiceInterfaceRegistAliveTaskId);
        ThrowError(resultRegistAliveTask);
        const ara::core::Result<void> resultReportAliveStatus = skeletonAdapter->InitializeMethod<ReportAliveStatusOutput>(methods::HzHmServiceInterfaceReportAliveStatusId);
        ThrowError(resultReportAliveStatus);
        const ara::core::Result<void> resultUnRegistAliveTask = skeletonAdapter->InitializeMethod<ara::core::Future<UnRegistAliveTaskOutput>>(methods::HzHmServiceInterfaceUnRegistAliveTaskId);
        ThrowError(resultUnRegistAliveTask);
        const ara::core::Result<void> resultRegistDeadlineTask = skeletonAdapter->InitializeMethod<ara::core::Future<RegistDeadlineTaskOutput>>(methods::HzHmServiceInterfaceRegistDeadlineTaskId);
        ThrowError(resultRegistDeadlineTask);
        const ara::core::Result<void> resultReportDeadlineStatus = skeletonAdapter->InitializeMethod<ReportDeadlineStatusOutput>(methods::HzHmServiceInterfaceReportDeadlineStatusId);
        ThrowError(resultReportDeadlineStatus);
        const ara::core::Result<void> resultUnRegistDeadlineTask = skeletonAdapter->InitializeMethod<ara::core::Future<UnRegistDeadlineTaskOutput>>(methods::HzHmServiceInterfaceUnRegistDeadlineTaskId);
        ThrowError(resultUnRegistDeadlineTask);
        const ara::core::Result<void> resultRegistLogicTask = skeletonAdapter->InitializeMethod<ara::core::Future<RegistLogicTaskOutput>>(methods::HzHmServiceInterfaceRegistLogicTaskId);
        ThrowError(resultRegistLogicTask);
        const ara::core::Result<void> resultReportLogicCheckpoint = skeletonAdapter->InitializeMethod<ara::core::Future<ReportLogicCheckpointOutput>>(methods::HzHmServiceInterfaceReportLogicCheckpointId);
        ThrowError(resultReportLogicCheckpoint);
        const ara::core::Result<void> resultUnRegistLogicTask = skeletonAdapter->InitializeMethod<ara::core::Future<UnRegistLogicTaskOutput>>(methods::HzHmServiceInterfaceUnRegistLogicTaskId);
        ThrowError(resultUnRegistLogicTask);
        const ara::core::Result<void> resultReportProcAliveCheckpoint = skeletonAdapter->InitializeMethod<ara::core::Future<ReportProcAliveCheckpointOutput>>(methods::HzHmServiceInterfaceReportProcAliveCheckpointId);
        ThrowError(resultReportProcAliveCheckpoint);
    }

    HzHmServiceInterfaceSkeleton& operator=(const HzHmServiceInterfaceSkeleton&) = delete;

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
    using RegistAliveTaskOutput = hozon::hm::methods::RegistAliveTask::Output;
    
    using ReportAliveStatusOutput = void;
    
    using UnRegistAliveTaskOutput = hozon::hm::methods::UnRegistAliveTask::Output;
    
    using RegistDeadlineTaskOutput = hozon::hm::methods::RegistDeadlineTask::Output;
    
    using ReportDeadlineStatusOutput = void;
    
    using UnRegistDeadlineTaskOutput = hozon::hm::methods::UnRegistDeadlineTask::Output;
    
    using RegistLogicTaskOutput = hozon::hm::methods::RegistLogicTask::Output;
    
    using ReportLogicCheckpointOutput = hozon::hm::methods::ReportLogicCheckpoint::Output;
    
    using UnRegistLogicTaskOutput = hozon::hm::methods::UnRegistLogicTask::Output;
    
    using ReportProcAliveCheckpointOutput = hozon::hm::methods::ReportProcAliveCheckpoint::Output;
    
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
    explicit HzHmServiceInterfaceSkeleton(const ara::com::InstanceIdentifier& instanceId,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        : skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::hm::HzHmServiceInterface::ServiceIdentifier, instanceId, mode)),
          RegistAliveTaskHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceRegistAliveTaskId),
          ReportAliveStatusHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceReportAliveStatusId),
          UnRegistAliveTaskHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceUnRegistAliveTaskId),
          RegistDeadlineTaskHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceRegistDeadlineTaskId),
          ReportDeadlineStatusHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceReportDeadlineStatusId),
          UnRegistDeadlineTaskHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceUnRegistDeadlineTaskId),
          RegistLogicTaskHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceRegistLogicTaskId),
          ReportLogicCheckpointHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceReportLogicCheckpointId),
          UnRegistLogicTaskHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceUnRegistLogicTaskId),
          ReportProcAliveCheckpointHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceReportProcAliveCheckpointId){
        ConstructSkeleton(mode);
    }

    explicit HzHmServiceInterfaceSkeleton(const ara::core::InstanceSpecifier& instanceSpec,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::hm::HzHmServiceInterface::ServiceIdentifier, instanceSpec, mode)),
          RegistAliveTaskHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceRegistAliveTaskId),
          ReportAliveStatusHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceReportAliveStatusId),
          UnRegistAliveTaskHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceUnRegistAliveTaskId),
          RegistDeadlineTaskHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceRegistDeadlineTaskId),
          ReportDeadlineStatusHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceReportDeadlineStatusId),
          UnRegistDeadlineTaskHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceUnRegistDeadlineTaskId),
          RegistLogicTaskHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceRegistLogicTaskId),
          ReportLogicCheckpointHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceReportLogicCheckpointId),
          UnRegistLogicTaskHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceUnRegistLogicTaskId),
          ReportProcAliveCheckpointHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceReportProcAliveCheckpointId){
        ConstructSkeleton(mode);
    }

    explicit HzHmServiceInterfaceSkeleton(const ara::com::InstanceIdentifierContainer instanceContainer,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::hm::HzHmServiceInterface::ServiceIdentifier, instanceContainer, mode)),
          RegistAliveTaskHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceRegistAliveTaskId),
          ReportAliveStatusHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceReportAliveStatusId),
          UnRegistAliveTaskHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceUnRegistAliveTaskId),
          RegistDeadlineTaskHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceRegistDeadlineTaskId),
          ReportDeadlineStatusHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceReportDeadlineStatusId),
          UnRegistDeadlineTaskHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceUnRegistDeadlineTaskId),
          RegistLogicTaskHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceRegistLogicTaskId),
          ReportLogicCheckpointHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceReportLogicCheckpointId),
          UnRegistLogicTaskHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceUnRegistLogicTaskId),
          ReportProcAliveCheckpointHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceReportProcAliveCheckpointId){
        ConstructSkeleton(mode);
    }

    HzHmServiceInterfaceSkeleton(const HzHmServiceInterfaceSkeleton&) = delete;

    HzHmServiceInterfaceSkeleton(HzHmServiceInterfaceSkeleton&&) = default;
    HzHmServiceInterfaceSkeleton& operator=(HzHmServiceInterfaceSkeleton&&) = default;
    HzHmServiceInterfaceSkeleton(ConstructionToken&& token) noexcept 
        : skeletonAdapter(token.GetSkeletonAdapter()),
          RegistAliveTaskHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceRegistAliveTaskId),
          ReportAliveStatusHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceReportAliveStatusId),
          UnRegistAliveTaskHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceUnRegistAliveTaskId),
          RegistDeadlineTaskHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceRegistDeadlineTaskId),
          ReportDeadlineStatusHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceReportDeadlineStatusId),
          UnRegistDeadlineTaskHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceUnRegistDeadlineTaskId),
          RegistLogicTaskHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceRegistLogicTaskId),
          ReportLogicCheckpointHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceReportLogicCheckpointId),
          UnRegistLogicTaskHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceUnRegistLogicTaskId),
          ReportProcAliveCheckpointHandle(skeletonAdapter->GetSkeleton(), methods::HzHmServiceInterfaceReportProcAliveCheckpointId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifier instanceId,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::hm::HzHmServiceInterface::ServiceIdentifier, instanceId, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::core::InstanceSpecifier instanceSpec,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::hm::HzHmServiceInterface::ServiceIdentifier, instanceSpec, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifierContainer instanceIdContainer,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::hm::HzHmServiceInterface::ServiceIdentifier, instanceIdContainer, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> PreConstructResult(
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter>& preSkeletonAdapter, const ara::com::MethodCallProcessingMode mode)
    {
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            if (mode == ara::com::MethodCallProcessingMode::kEvent) {
                if(!preSkeletonAdapter->SetMethodThreadNumber(preSkeletonAdapter->GetMethodThreadNumber(10U), 1024U)) {
                    result = false;
                    ara::core::ErrorCode errorcode(ara::com::ComErrc::kNetworkBindingFailure);
                    initResult.EmplaceError(errorcode);
                    break;
                }
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<RegistAliveTaskOutput>>(methods::HzHmServiceInterfaceRegistAliveTaskId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ReportAliveStatusOutput>(methods::HzHmServiceInterfaceReportAliveStatusId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<UnRegistAliveTaskOutput>>(methods::HzHmServiceInterfaceUnRegistAliveTaskId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<RegistDeadlineTaskOutput>>(methods::HzHmServiceInterfaceRegistDeadlineTaskId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ReportDeadlineStatusOutput>(methods::HzHmServiceInterfaceReportDeadlineStatusId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<UnRegistDeadlineTaskOutput>>(methods::HzHmServiceInterfaceUnRegistDeadlineTaskId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<RegistLogicTaskOutput>>(methods::HzHmServiceInterfaceRegistLogicTaskId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<ReportLogicCheckpointOutput>>(methods::HzHmServiceInterfaceReportLogicCheckpointId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<UnRegistLogicTaskOutput>>(methods::HzHmServiceInterfaceUnRegistLogicTaskId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<ReportProcAliveCheckpointOutput>>(methods::HzHmServiceInterfaceReportProcAliveCheckpointId);
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

    virtual ~HzHmServiceInterfaceSkeleton()
    {
        StopOfferService();
    }

    void OfferService()
    {
        skeletonAdapter->RegisterE2EErrorHandler(&HzHmServiceInterfaceSkeleton::E2EErrorHandler, *this);
        skeletonAdapter->RegisterMethod(&HzHmServiceInterfaceSkeleton::RegistAliveTask, *this, methods::HzHmServiceInterfaceRegistAliveTaskId);
        skeletonAdapter->RegisterMethod(&HzHmServiceInterfaceSkeleton::ReportAliveStatus, *this, methods::HzHmServiceInterfaceReportAliveStatusId);
        skeletonAdapter->RegisterMethod(&HzHmServiceInterfaceSkeleton::UnRegistAliveTask, *this, methods::HzHmServiceInterfaceUnRegistAliveTaskId);
        skeletonAdapter->RegisterMethod(&HzHmServiceInterfaceSkeleton::RegistDeadlineTask, *this, methods::HzHmServiceInterfaceRegistDeadlineTaskId);
        skeletonAdapter->RegisterMethod(&HzHmServiceInterfaceSkeleton::ReportDeadlineStatus, *this, methods::HzHmServiceInterfaceReportDeadlineStatusId);
        skeletonAdapter->RegisterMethod(&HzHmServiceInterfaceSkeleton::UnRegistDeadlineTask, *this, methods::HzHmServiceInterfaceUnRegistDeadlineTaskId);
        skeletonAdapter->RegisterMethod(&HzHmServiceInterfaceSkeleton::RegistLogicTask, *this, methods::HzHmServiceInterfaceRegistLogicTaskId);
        skeletonAdapter->RegisterMethod(&HzHmServiceInterfaceSkeleton::ReportLogicCheckpoint, *this, methods::HzHmServiceInterfaceReportLogicCheckpointId);
        skeletonAdapter->RegisterMethod(&HzHmServiceInterfaceSkeleton::UnRegistLogicTask, *this, methods::HzHmServiceInterfaceUnRegistLogicTaskId);
        skeletonAdapter->RegisterMethod(&HzHmServiceInterfaceSkeleton::ReportProcAliveCheckpoint, *this, methods::HzHmServiceInterfaceReportProcAliveCheckpointId);
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
    virtual ara::core::Future<RegistAliveTaskOutput> RegistAliveTask(const ::String& processName, const ::uint32_t& checkpointId, const ::uint32_t& periodMs, const ::uint32_t& minIndication, const ::uint32_t& maxIndication) = 0;
    virtual ReportAliveStatusOutput ReportAliveStatus(const ::String& processName, const ::uint32_t& checkpointId, const ::uint8_t& aliveStatus) = 0;
    virtual ara::core::Future<UnRegistAliveTaskOutput> UnRegistAliveTask(const ::String& processName, const ::uint32_t& checkpointId) = 0;
    virtual ara::core::Future<RegistDeadlineTaskOutput> RegistDeadlineTask(const ::String& processName, const ::hozon::hm::Transition& transition, const ::uint32_t& deadlineMinMs, const ::uint32_t& deadlineMaxMs) = 0;
    virtual ReportDeadlineStatusOutput ReportDeadlineStatus(const ::String& processName, const ::hozon::hm::Transition& transition, const ::uint8_t& deadlineStatus) = 0;
    virtual ara::core::Future<UnRegistDeadlineTaskOutput> UnRegistDeadlineTask(const ::String& processName, const ::hozon::hm::Transition& transition) = 0;
    virtual ara::core::Future<RegistLogicTaskOutput> RegistLogicTask(const ::String& processName, const ::hozon::hm::Transitions& checkpointIds) = 0;
    virtual ara::core::Future<ReportLogicCheckpointOutput> ReportLogicCheckpoint(const ::String& processName, const ::uint32_t& checkpointId) = 0;
    virtual ara::core::Future<UnRegistLogicTaskOutput> UnRegistLogicTask(const ::String& processName, const ::hozon::hm::Transitions& checkpointIds) = 0;
    virtual ara::core::Future<ReportProcAliveCheckpointOutput> ReportProcAliveCheckpoint(const ::String& processName, const ::uint32_t& checkpointId) = 0;
    virtual void E2EErrorHandler(ara::com::e2e::E2EErrorCode, ara::com::e2e::DataID, ara::com::e2e::MessageCounter){}

    methods::RegistAliveTaskHandle RegistAliveTaskHandle;
    methods::ReportAliveStatusHandle ReportAliveStatusHandle;
    methods::UnRegistAliveTaskHandle UnRegistAliveTaskHandle;
    methods::RegistDeadlineTaskHandle RegistDeadlineTaskHandle;
    methods::ReportDeadlineStatusHandle ReportDeadlineStatusHandle;
    methods::UnRegistDeadlineTaskHandle UnRegistDeadlineTaskHandle;
    methods::RegistLogicTaskHandle RegistLogicTaskHandle;
    methods::ReportLogicCheckpointHandle ReportLogicCheckpointHandle;
    methods::UnRegistLogicTaskHandle UnRegistLogicTaskHandle;
    methods::ReportProcAliveCheckpointHandle ReportProcAliveCheckpointHandle;
};
} // namespace skeleton
} // namespace hm
} // namespace hozon

#endif // HOZON_HM_HZHMSERVICEINTERFACE_SKELETON_H
