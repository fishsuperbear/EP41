/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SM_HZSMADAPTORSERVICEINTERFACE_SKELETON_H
#define HOZON_SM_HZSMADAPTORSERVICEINTERFACE_SKELETON_H

#include "ara/com/internal/skeleton/skeleton_adapter.h"
#include "ara/com/internal/skeleton/event_adapter.h"
#include "ara/com/internal/skeleton/field_adapter.h"
#include "ara/com/internal/skeleton/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/sm/hzsmadaptorserviceinterface_common.h"
#include <cstdint>

namespace hozon {
namespace sm {
namespace skeleton {
namespace events
{
}

namespace methods
{
    using FuncGroupStateChangeHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using MultiFuncGroupStateChangeHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using QueryFuncFroupStateHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using RestartProcByNameHandle = ara::com::internal::skeleton::method::MethodAdapter;
    static constexpr ara::com::internal::EntityId HzSmAdaptorServiceInterfaceFuncGroupStateChangeId = 15833U; //FuncGroupStateChange_method_hash
    static constexpr ara::com::internal::EntityId HzSmAdaptorServiceInterfaceMultiFuncGroupStateChangeId = 62434U; //MultiFuncGroupStateChange_method_hash
    static constexpr ara::com::internal::EntityId HzSmAdaptorServiceInterfaceQueryFuncFroupStateId = 46362U; //QueryFuncFroupState_method_hash
    static constexpr ara::com::internal::EntityId HzSmAdaptorServiceInterfaceRestartProcByNameId = 63703U; //RestartProcByName_method_hash
}

namespace fields
{
    using MachineState = ara::com::internal::skeleton::field::FieldAdapter<::String>;
    static constexpr ara::com::internal::EntityId HzSmAdaptorServiceInterfaceMachineStateId = 21022U; //MachineState_field_hash
}

class HzSmAdaptorServiceInterfaceSkeleton {
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
        const ara::core::Result<void> resultFuncGroupStateChange = skeletonAdapter->InitializeMethod<ara::core::Future<FuncGroupStateChangeOutput>>(methods::HzSmAdaptorServiceInterfaceFuncGroupStateChangeId);
        ThrowError(resultFuncGroupStateChange);
        const ara::core::Result<void> resultMultiFuncGroupStateChange = skeletonAdapter->InitializeMethod<ara::core::Future<MultiFuncGroupStateChangeOutput>>(methods::HzSmAdaptorServiceInterfaceMultiFuncGroupStateChangeId);
        ThrowError(resultMultiFuncGroupStateChange);
        const ara::core::Result<void> resultQueryFuncFroupState = skeletonAdapter->InitializeMethod<ara::core::Future<QueryFuncFroupStateOutput>>(methods::HzSmAdaptorServiceInterfaceQueryFuncFroupStateId);
        ThrowError(resultQueryFuncFroupState);
        const ara::core::Result<void> resultRestartProcByName = skeletonAdapter->InitializeMethod<RestartProcByNameOutput>(methods::HzSmAdaptorServiceInterfaceRestartProcByNameId);
        ThrowError(resultRestartProcByName);
        const ara::core::Result<void> resultMachineState = skeletonAdapter->InitializeField<::String>(MachineState);
        ThrowError(resultMachineState);
    }

    HzSmAdaptorServiceInterfaceSkeleton& operator=(const HzSmAdaptorServiceInterfaceSkeleton&) = delete;

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
    using FuncGroupStateChangeOutput = hozon::sm::methods::FuncGroupStateChange::Output;
    
    using MultiFuncGroupStateChangeOutput = hozon::sm::methods::MultiFuncGroupStateChange::Output;
    
    using QueryFuncFroupStateOutput = hozon::sm::methods::QueryFuncFroupState::Output;
    
    using RestartProcByNameOutput = void;
    
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
        fields::MachineState MachineState;
    };
    explicit HzSmAdaptorServiceInterfaceSkeleton(const ara::com::InstanceIdentifier& instanceId,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        : skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::sm::HzSmAdaptorServiceInterface::ServiceIdentifier, instanceId, mode)),
          FuncGroupStateChangeHandle(skeletonAdapter->GetSkeleton(), methods::HzSmAdaptorServiceInterfaceFuncGroupStateChangeId),
          MultiFuncGroupStateChangeHandle(skeletonAdapter->GetSkeleton(), methods::HzSmAdaptorServiceInterfaceMultiFuncGroupStateChangeId),
          QueryFuncFroupStateHandle(skeletonAdapter->GetSkeleton(), methods::HzSmAdaptorServiceInterfaceQueryFuncFroupStateId),
          RestartProcByNameHandle(skeletonAdapter->GetSkeleton(), methods::HzSmAdaptorServiceInterfaceRestartProcByNameId),
          MachineState(skeletonAdapter->GetSkeleton(), fields::HzSmAdaptorServiceInterfaceMachineStateId) {
        ConstructSkeleton(mode);
    }

    explicit HzSmAdaptorServiceInterfaceSkeleton(const ara::core::InstanceSpecifier& instanceSpec,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::sm::HzSmAdaptorServiceInterface::ServiceIdentifier, instanceSpec, mode)),
          FuncGroupStateChangeHandle(skeletonAdapter->GetSkeleton(), methods::HzSmAdaptorServiceInterfaceFuncGroupStateChangeId),
          MultiFuncGroupStateChangeHandle(skeletonAdapter->GetSkeleton(), methods::HzSmAdaptorServiceInterfaceMultiFuncGroupStateChangeId),
          QueryFuncFroupStateHandle(skeletonAdapter->GetSkeleton(), methods::HzSmAdaptorServiceInterfaceQueryFuncFroupStateId),
          RestartProcByNameHandle(skeletonAdapter->GetSkeleton(), methods::HzSmAdaptorServiceInterfaceRestartProcByNameId),
          MachineState(skeletonAdapter->GetSkeleton(), fields::HzSmAdaptorServiceInterfaceMachineStateId) {
        ConstructSkeleton(mode);
    }

    explicit HzSmAdaptorServiceInterfaceSkeleton(const ara::com::InstanceIdentifierContainer instanceContainer,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::sm::HzSmAdaptorServiceInterface::ServiceIdentifier, instanceContainer, mode)),
          FuncGroupStateChangeHandle(skeletonAdapter->GetSkeleton(), methods::HzSmAdaptorServiceInterfaceFuncGroupStateChangeId),
          MultiFuncGroupStateChangeHandle(skeletonAdapter->GetSkeleton(), methods::HzSmAdaptorServiceInterfaceMultiFuncGroupStateChangeId),
          QueryFuncFroupStateHandle(skeletonAdapter->GetSkeleton(), methods::HzSmAdaptorServiceInterfaceQueryFuncFroupStateId),
          RestartProcByNameHandle(skeletonAdapter->GetSkeleton(), methods::HzSmAdaptorServiceInterfaceRestartProcByNameId),
          MachineState(skeletonAdapter->GetSkeleton(), fields::HzSmAdaptorServiceInterfaceMachineStateId) {
        ConstructSkeleton(mode);
    }

    HzSmAdaptorServiceInterfaceSkeleton(const HzSmAdaptorServiceInterfaceSkeleton&) = delete;

    HzSmAdaptorServiceInterfaceSkeleton(HzSmAdaptorServiceInterfaceSkeleton&&) = default;
    HzSmAdaptorServiceInterfaceSkeleton& operator=(HzSmAdaptorServiceInterfaceSkeleton&&) = default;
    HzSmAdaptorServiceInterfaceSkeleton(ConstructionToken&& token) noexcept 
        : skeletonAdapter(token.GetSkeletonAdapter()),
          FuncGroupStateChangeHandle(skeletonAdapter->GetSkeleton(), methods::HzSmAdaptorServiceInterfaceFuncGroupStateChangeId),
          MultiFuncGroupStateChangeHandle(skeletonAdapter->GetSkeleton(), methods::HzSmAdaptorServiceInterfaceMultiFuncGroupStateChangeId),
          QueryFuncFroupStateHandle(skeletonAdapter->GetSkeleton(), methods::HzSmAdaptorServiceInterfaceQueryFuncFroupStateId),
          RestartProcByNameHandle(skeletonAdapter->GetSkeleton(), methods::HzSmAdaptorServiceInterfaceRestartProcByNameId),
          MachineState(skeletonAdapter->GetSkeleton(), fields::HzSmAdaptorServiceInterfaceMachineStateId) {
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifier instanceId,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::sm::HzSmAdaptorServiceInterface::ServiceIdentifier, instanceId, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::core::InstanceSpecifier instanceSpec,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::sm::HzSmAdaptorServiceInterface::ServiceIdentifier, instanceSpec, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifierContainer instanceIdContainer,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::sm::HzSmAdaptorServiceInterface::ServiceIdentifier, instanceIdContainer, mode);
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
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<FuncGroupStateChangeOutput>>(methods::HzSmAdaptorServiceInterfaceFuncGroupStateChangeId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<MultiFuncGroupStateChangeOutput>>(methods::HzSmAdaptorServiceInterfaceMultiFuncGroupStateChangeId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<QueryFuncFroupStateOutput>>(methods::HzSmAdaptorServiceInterfaceQueryFuncFroupStateId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<RestartProcByNameOutput>(methods::HzSmAdaptorServiceInterfaceRestartProcByNameId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            fields::MachineState MachineState(preSkeletonAdapter->GetSkeleton(), fields::HzSmAdaptorServiceInterfaceMachineStateId);
            initResult = preSkeletonAdapter->InitializeField<::String>(MachineState);
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

    virtual ~HzSmAdaptorServiceInterfaceSkeleton()
    {
        StopOfferService();
    }

    void OfferService()
    {
        skeletonAdapter->RegisterE2EErrorHandler(&HzSmAdaptorServiceInterfaceSkeleton::E2EErrorHandler, *this);
        skeletonAdapter->RegisterMethod(&HzSmAdaptorServiceInterfaceSkeleton::FuncGroupStateChange, *this, methods::HzSmAdaptorServiceInterfaceFuncGroupStateChangeId);
        skeletonAdapter->RegisterMethod(&HzSmAdaptorServiceInterfaceSkeleton::MultiFuncGroupStateChange, *this, methods::HzSmAdaptorServiceInterfaceMultiFuncGroupStateChangeId);
        skeletonAdapter->RegisterMethod(&HzSmAdaptorServiceInterfaceSkeleton::QueryFuncFroupState, *this, methods::HzSmAdaptorServiceInterfaceQueryFuncFroupStateId);
        skeletonAdapter->RegisterMethod(&HzSmAdaptorServiceInterfaceSkeleton::RestartProcByName, *this, methods::HzSmAdaptorServiceInterfaceRestartProcByNameId);
        MachineState.VerifyValidity();
        skeletonAdapter->OfferService();
    }
    void StopOfferService()
    {
        skeletonAdapter->StopOfferService();
        MachineState.ResetInitState();
    }
    ara::core::Future<bool> ProcessNextMethodCall()
    {
        return skeletonAdapter->ProcessNextMethodCall();
    }
    bool SetMethodThreadNumber(const std::uint16_t& number, const std::uint16_t& queueSize)
    {
        return skeletonAdapter->SetMethodThreadNumber(number, queueSize);
    }
    virtual ara::core::Future<FuncGroupStateChangeOutput> FuncGroupStateChange(const ::hozon::sm::FGStateChange& stateChange) = 0;
    virtual ara::core::Future<MultiFuncGroupStateChangeOutput> MultiFuncGroupStateChange(const ::hozon::sm::FGStateChangeVector& stateChanges) = 0;
    virtual ara::core::Future<QueryFuncFroupStateOutput> QueryFuncFroupState(const ::String& fgName) = 0;
    virtual RestartProcByNameOutput RestartProcByName(const ::String& procName) = 0;
    virtual void E2EErrorHandler(ara::com::e2e::E2EErrorCode, ara::com::e2e::DataID, ara::com::e2e::MessageCounter){}

    methods::FuncGroupStateChangeHandle FuncGroupStateChangeHandle;
    methods::MultiFuncGroupStateChangeHandle MultiFuncGroupStateChangeHandle;
    methods::QueryFuncFroupStateHandle QueryFuncFroupStateHandle;
    methods::RestartProcByNameHandle RestartProcByNameHandle;
    fields::MachineState MachineState;
};
} // namespace skeleton
} // namespace sm
} // namespace hozon

#endif // HOZON_SM_HZSMADAPTORSERVICEINTERFACE_SKELETON_H
