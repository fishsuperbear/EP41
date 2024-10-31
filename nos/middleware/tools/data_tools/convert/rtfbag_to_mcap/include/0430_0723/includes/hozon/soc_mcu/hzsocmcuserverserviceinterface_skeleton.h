/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_HZSOCMCUSERVERSERVICEINTERFACE_SKELETON_H
#define HOZON_SOC_MCU_HZSOCMCUSERVERSERVICEINTERFACE_SKELETON_H

#include "ara/com/internal/skeleton/skeleton_adapter.h"
#include "ara/com/internal/skeleton/event_adapter.h"
#include "ara/com/internal/skeleton/field_adapter.h"
#include "ara/com/internal/skeleton/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/soc_mcu/hzsocmcuserverserviceinterface_common.h"
#include <cstdint>

namespace hozon {
namespace soc_mcu {
namespace skeleton {
namespace events
{
    using TrajDataEvent = ara::com::internal::skeleton::event::EventAdapter<::hozon::soc_mcu::struct_soc_mcu_array>;
    using PoseDataEvent = ara::com::internal::skeleton::event::EventAdapter<::hozon::soc_mcu::struct_soc_mcu_array>;
    using SnsrFsnLaneDateEvent = ara::com::internal::skeleton::event::EventAdapter<::hozon::soc_mcu::struct_soc_mcu_array>;
    using SnsrFsnObjDataEvent = ara::com::internal::skeleton::event::EventAdapter<::hozon::soc_mcu::struct_soc_mcu_array>;
    using TrajDataEvent1 = ara::com::internal::skeleton::event::EventAdapter<::hozon::soc_mcu::EgoTrajectoryFrame_soc_mcu>;
    using PoseDataEvent1 = ara::com::internal::skeleton::event::EventAdapter<::hozon::soc_mcu::LocationFrame_soc_mcu>;
    using SnsrFsnLaneDateEvent1 = ara::com::internal::skeleton::event::EventAdapter<::hozon::soc_mcu::LaneLineArray_soc_mcu>;
    using SnsrFsnObjDataEvent1 = ara::com::internal::skeleton::event::EventAdapter<::hozon::soc_mcu::ObjectFusionFrame_soc_mcu>;
    using FreeSpaceDataEvent1 = ara::com::internal::skeleton::event::EventAdapter<::hozon::freespace::FreeSpaceFrame>;
    using UssDataEvent1 = ara::com::internal::skeleton::event::EventAdapter<::hozon::soc_mcu::UssInfo_soc_mcu>;
    static constexpr ara::com::internal::EntityId HzSocMcuServerServiceInterfaceTrajDataEventId = 48553U; //TrajDataEvent_event_hash
    static constexpr ara::com::internal::EntityId HzSocMcuServerServiceInterfacePoseDataEventId = 50505U; //PoseDataEvent_event_hash
    static constexpr ara::com::internal::EntityId HzSocMcuServerServiceInterfaceSnsrFsnLaneDateEventId = 46863U; //SnsrFsnLaneDateEvent_event_hash
    static constexpr ara::com::internal::EntityId HzSocMcuServerServiceInterfaceSnsrFsnObjDataEventId = 53207U; //SnsrFsnObjDataEvent_event_hash
    static constexpr ara::com::internal::EntityId HzSocMcuServerServiceInterfaceTrajDataEvent1Id = 48735U; //TrajDataEvent1_event_hash
    static constexpr ara::com::internal::EntityId HzSocMcuServerServiceInterfacePoseDataEvent1Id = 30021U; //PoseDataEvent1_event_hash
    static constexpr ara::com::internal::EntityId HzSocMcuServerServiceInterfaceSnsrFsnLaneDateEvent1Id = 10487U; //SnsrFsnLaneDateEvent1_event_hash
    static constexpr ara::com::internal::EntityId HzSocMcuServerServiceInterfaceSnsrFsnObjDataEvent1Id = 18926U; //SnsrFsnObjDataEvent1_event_hash
    static constexpr ara::com::internal::EntityId HzSocMcuServerServiceInterfaceFreeSpaceDataEvent1Id = 33116U; //FreeSpaceDataEvent1_event_hash
    static constexpr ara::com::internal::EntityId HzSocMcuServerServiceInterfaceUssDataEvent1Id = 60981U; //UssDataEvent1_event_hash
}

namespace methods
{
    using TestaddHandle = ara::com::internal::skeleton::method::MethodAdapter;
    static constexpr ara::com::internal::EntityId HzSocMcuServerServiceInterfaceTestaddId = 22266U; //Testadd_method_hash
}

namespace fields
{
    using Test_Field = ara::com::internal::skeleton::field::FieldAdapter<::UInt32>;
    static constexpr ara::com::internal::EntityId HzSocMcuServerServiceInterfaceTest_FieldId = 17018U; //Test_Field_field_hash
    static constexpr ara::com::internal::EntityId HzSocMcuServerServiceInterfaceTest_FieldSetterId = 35747U; //Test_Field_setter_hash
    static constexpr ara::com::internal::EntityId HzSocMcuServerServiceInterfaceTest_FieldGetterId = 12623U; //Test_Field_getter_hash
}

class HzSocMcuServerServiceInterfaceSkeleton {
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
        const ara::core::Result<void> resultTrajDataEvent = skeletonAdapter->InitializeEvent(TrajDataEvent);
        ThrowError(resultTrajDataEvent);
        const ara::core::Result<void> resultPoseDataEvent = skeletonAdapter->InitializeEvent(PoseDataEvent);
        ThrowError(resultPoseDataEvent);
        const ara::core::Result<void> resultSnsrFsnLaneDateEvent = skeletonAdapter->InitializeEvent(SnsrFsnLaneDateEvent);
        ThrowError(resultSnsrFsnLaneDateEvent);
        const ara::core::Result<void> resultSnsrFsnObjDataEvent = skeletonAdapter->InitializeEvent(SnsrFsnObjDataEvent);
        ThrowError(resultSnsrFsnObjDataEvent);
        const ara::core::Result<void> resultTrajDataEvent1 = skeletonAdapter->InitializeEvent(TrajDataEvent1);
        ThrowError(resultTrajDataEvent1);
        const ara::core::Result<void> resultPoseDataEvent1 = skeletonAdapter->InitializeEvent(PoseDataEvent1);
        ThrowError(resultPoseDataEvent1);
        const ara::core::Result<void> resultSnsrFsnLaneDateEvent1 = skeletonAdapter->InitializeEvent(SnsrFsnLaneDateEvent1);
        ThrowError(resultSnsrFsnLaneDateEvent1);
        const ara::core::Result<void> resultSnsrFsnObjDataEvent1 = skeletonAdapter->InitializeEvent(SnsrFsnObjDataEvent1);
        ThrowError(resultSnsrFsnObjDataEvent1);
        const ara::core::Result<void> resultFreeSpaceDataEvent1 = skeletonAdapter->InitializeEvent(FreeSpaceDataEvent1);
        ThrowError(resultFreeSpaceDataEvent1);
        const ara::core::Result<void> resultUssDataEvent1 = skeletonAdapter->InitializeEvent(UssDataEvent1);
        ThrowError(resultUssDataEvent1);
        const ara::core::Result<void> resultTestadd = skeletonAdapter->InitializeMethod<ara::core::Future<TestaddOutput>>(methods::HzSocMcuServerServiceInterfaceTestaddId);
        ThrowError(resultTestadd);
        Test_Field.SetSetterEntityId(fields::HzSocMcuServerServiceInterfaceTest_FieldSetterId);
        Test_Field.SetGetterEntityId(fields::HzSocMcuServerServiceInterfaceTest_FieldGetterId);
        const ara::core::Result<void> resultTest_Field = skeletonAdapter->InitializeField<::UInt32>(Test_Field);
        ThrowError(resultTest_Field);
    }

    HzSocMcuServerServiceInterfaceSkeleton& operator=(const HzSocMcuServerServiceInterfaceSkeleton&) = delete;

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
    using TestaddOutput = hozon::soc_mcu::methods::Testadd::Output;
    
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
        fields::Test_Field Test_Field;
    };
    explicit HzSocMcuServerServiceInterfaceSkeleton(const ara::com::InstanceIdentifier& instanceId,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        : skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::soc_mcu::HzSocMcuServerServiceInterface::ServiceIdentifier, instanceId, mode)),
          TrajDataEvent(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceTrajDataEventId),
          PoseDataEvent(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfacePoseDataEventId),
          SnsrFsnLaneDateEvent(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceSnsrFsnLaneDateEventId),
          SnsrFsnObjDataEvent(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceSnsrFsnObjDataEventId),
          TrajDataEvent1(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceTrajDataEvent1Id),
          PoseDataEvent1(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfacePoseDataEvent1Id),
          SnsrFsnLaneDateEvent1(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceSnsrFsnLaneDateEvent1Id),
          SnsrFsnObjDataEvent1(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceSnsrFsnObjDataEvent1Id),
          FreeSpaceDataEvent1(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceFreeSpaceDataEvent1Id),
          UssDataEvent1(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceUssDataEvent1Id),
          TestaddHandle(skeletonAdapter->GetSkeleton(), methods::HzSocMcuServerServiceInterfaceTestaddId),
          Test_Field(skeletonAdapter->GetSkeleton(), fields::HzSocMcuServerServiceInterfaceTest_FieldId) {
        ConstructSkeleton(mode);
    }

    explicit HzSocMcuServerServiceInterfaceSkeleton(const ara::core::InstanceSpecifier& instanceSpec,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::soc_mcu::HzSocMcuServerServiceInterface::ServiceIdentifier, instanceSpec, mode)),
          TrajDataEvent(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceTrajDataEventId),
          PoseDataEvent(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfacePoseDataEventId),
          SnsrFsnLaneDateEvent(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceSnsrFsnLaneDateEventId),
          SnsrFsnObjDataEvent(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceSnsrFsnObjDataEventId),
          TrajDataEvent1(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceTrajDataEvent1Id),
          PoseDataEvent1(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfacePoseDataEvent1Id),
          SnsrFsnLaneDateEvent1(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceSnsrFsnLaneDateEvent1Id),
          SnsrFsnObjDataEvent1(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceSnsrFsnObjDataEvent1Id),
          FreeSpaceDataEvent1(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceFreeSpaceDataEvent1Id),
          UssDataEvent1(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceUssDataEvent1Id),
          TestaddHandle(skeletonAdapter->GetSkeleton(), methods::HzSocMcuServerServiceInterfaceTestaddId),
          Test_Field(skeletonAdapter->GetSkeleton(), fields::HzSocMcuServerServiceInterfaceTest_FieldId) {
        ConstructSkeleton(mode);
    }

    explicit HzSocMcuServerServiceInterfaceSkeleton(const ara::com::InstanceIdentifierContainer instanceContainer,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::soc_mcu::HzSocMcuServerServiceInterface::ServiceIdentifier, instanceContainer, mode)),
          TrajDataEvent(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceTrajDataEventId),
          PoseDataEvent(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfacePoseDataEventId),
          SnsrFsnLaneDateEvent(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceSnsrFsnLaneDateEventId),
          SnsrFsnObjDataEvent(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceSnsrFsnObjDataEventId),
          TrajDataEvent1(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceTrajDataEvent1Id),
          PoseDataEvent1(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfacePoseDataEvent1Id),
          SnsrFsnLaneDateEvent1(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceSnsrFsnLaneDateEvent1Id),
          SnsrFsnObjDataEvent1(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceSnsrFsnObjDataEvent1Id),
          FreeSpaceDataEvent1(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceFreeSpaceDataEvent1Id),
          UssDataEvent1(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceUssDataEvent1Id),
          TestaddHandle(skeletonAdapter->GetSkeleton(), methods::HzSocMcuServerServiceInterfaceTestaddId),
          Test_Field(skeletonAdapter->GetSkeleton(), fields::HzSocMcuServerServiceInterfaceTest_FieldId) {
        ConstructSkeleton(mode);
    }

    HzSocMcuServerServiceInterfaceSkeleton(const HzSocMcuServerServiceInterfaceSkeleton&) = delete;

    HzSocMcuServerServiceInterfaceSkeleton(HzSocMcuServerServiceInterfaceSkeleton&&) = default;
    HzSocMcuServerServiceInterfaceSkeleton& operator=(HzSocMcuServerServiceInterfaceSkeleton&&) = default;
    HzSocMcuServerServiceInterfaceSkeleton(ConstructionToken&& token) noexcept 
        : skeletonAdapter(token.GetSkeletonAdapter()),
          TrajDataEvent(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceTrajDataEventId),
          PoseDataEvent(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfacePoseDataEventId),
          SnsrFsnLaneDateEvent(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceSnsrFsnLaneDateEventId),
          SnsrFsnObjDataEvent(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceSnsrFsnObjDataEventId),
          TrajDataEvent1(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceTrajDataEvent1Id),
          PoseDataEvent1(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfacePoseDataEvent1Id),
          SnsrFsnLaneDateEvent1(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceSnsrFsnLaneDateEvent1Id),
          SnsrFsnObjDataEvent1(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceSnsrFsnObjDataEvent1Id),
          FreeSpaceDataEvent1(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceFreeSpaceDataEvent1Id),
          UssDataEvent1(skeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceUssDataEvent1Id),
          TestaddHandle(skeletonAdapter->GetSkeleton(), methods::HzSocMcuServerServiceInterfaceTestaddId),
          Test_Field(skeletonAdapter->GetSkeleton(), fields::HzSocMcuServerServiceInterfaceTest_FieldId) {
        Test_Field.SetSetterEntityId(fields::HzSocMcuServerServiceInterfaceTest_FieldSetterId);
        Test_Field.SetGetterEntityId(fields::HzSocMcuServerServiceInterfaceTest_FieldGetterId);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifier instanceId,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::soc_mcu::HzSocMcuServerServiceInterface::ServiceIdentifier, instanceId, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::core::InstanceSpecifier instanceSpec,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::soc_mcu::HzSocMcuServerServiceInterface::ServiceIdentifier, instanceSpec, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifierContainer instanceIdContainer,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::soc_mcu::HzSocMcuServerServiceInterface::ServiceIdentifier, instanceIdContainer, mode);
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
            const events::TrajDataEvent TrajDataEvent(preSkeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceTrajDataEventId);
            initResult = preSkeletonAdapter->InitializeEvent(TrajDataEvent);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const events::PoseDataEvent PoseDataEvent(preSkeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfacePoseDataEventId);
            initResult = preSkeletonAdapter->InitializeEvent(PoseDataEvent);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const events::SnsrFsnLaneDateEvent SnsrFsnLaneDateEvent(preSkeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceSnsrFsnLaneDateEventId);
            initResult = preSkeletonAdapter->InitializeEvent(SnsrFsnLaneDateEvent);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const events::SnsrFsnObjDataEvent SnsrFsnObjDataEvent(preSkeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceSnsrFsnObjDataEventId);
            initResult = preSkeletonAdapter->InitializeEvent(SnsrFsnObjDataEvent);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const events::TrajDataEvent1 TrajDataEvent1(preSkeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceTrajDataEvent1Id);
            initResult = preSkeletonAdapter->InitializeEvent(TrajDataEvent1);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const events::PoseDataEvent1 PoseDataEvent1(preSkeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfacePoseDataEvent1Id);
            initResult = preSkeletonAdapter->InitializeEvent(PoseDataEvent1);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const events::SnsrFsnLaneDateEvent1 SnsrFsnLaneDateEvent1(preSkeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceSnsrFsnLaneDateEvent1Id);
            initResult = preSkeletonAdapter->InitializeEvent(SnsrFsnLaneDateEvent1);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const events::SnsrFsnObjDataEvent1 SnsrFsnObjDataEvent1(preSkeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceSnsrFsnObjDataEvent1Id);
            initResult = preSkeletonAdapter->InitializeEvent(SnsrFsnObjDataEvent1);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const events::FreeSpaceDataEvent1 FreeSpaceDataEvent1(preSkeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceFreeSpaceDataEvent1Id);
            initResult = preSkeletonAdapter->InitializeEvent(FreeSpaceDataEvent1);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const events::UssDataEvent1 UssDataEvent1(preSkeletonAdapter->GetSkeleton(), events::HzSocMcuServerServiceInterfaceUssDataEvent1Id);
            initResult = preSkeletonAdapter->InitializeEvent(UssDataEvent1);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<TestaddOutput>>(methods::HzSocMcuServerServiceInterfaceTestaddId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            fields::Test_Field Test_Field(preSkeletonAdapter->GetSkeleton(), fields::HzSocMcuServerServiceInterfaceTest_FieldId);
            Test_Field.SetSetterEntityId(fields::HzSocMcuServerServiceInterfaceTest_FieldSetterId);
            Test_Field.SetGetterEntityId(fields::HzSocMcuServerServiceInterfaceTest_FieldGetterId);
            initResult = preSkeletonAdapter->InitializeField<::UInt32>(Test_Field);
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

    virtual ~HzSocMcuServerServiceInterfaceSkeleton()
    {
        StopOfferService();
    }

    void OfferService()
    {
        skeletonAdapter->RegisterE2EErrorHandler(&HzSocMcuServerServiceInterfaceSkeleton::E2EErrorHandler, *this);
        skeletonAdapter->RegisterMethod(&HzSocMcuServerServiceInterfaceSkeleton::Testadd, *this, methods::HzSocMcuServerServiceInterfaceTestaddId);
        Test_Field.VerifyValidity();
        skeletonAdapter->OfferService();
    }
    void StopOfferService()
    {
        skeletonAdapter->StopOfferService();
        Test_Field.ResetInitState();
    }
    ara::core::Future<bool> ProcessNextMethodCall()
    {
        return skeletonAdapter->ProcessNextMethodCall();
    }
    bool SetMethodThreadNumber(const std::uint16_t& number, const std::uint16_t& queueSize)
    {
        return skeletonAdapter->SetMethodThreadNumber(number, queueSize);
    }
    virtual ara::core::Future<TestaddOutput> Testadd(const ::UInt8& parameter1_input, const ::UInt8& parameter2_in) = 0;
    virtual void E2EErrorHandler(ara::com::e2e::E2EErrorCode, ara::com::e2e::DataID, ara::com::e2e::MessageCounter){}

    events::TrajDataEvent TrajDataEvent;
    events::PoseDataEvent PoseDataEvent;
    events::SnsrFsnLaneDateEvent SnsrFsnLaneDateEvent;
    events::SnsrFsnObjDataEvent SnsrFsnObjDataEvent;
    events::TrajDataEvent1 TrajDataEvent1;
    events::PoseDataEvent1 PoseDataEvent1;
    events::SnsrFsnLaneDateEvent1 SnsrFsnLaneDateEvent1;
    events::SnsrFsnObjDataEvent1 SnsrFsnObjDataEvent1;
    events::FreeSpaceDataEvent1 FreeSpaceDataEvent1;
    events::UssDataEvent1 UssDataEvent1;
    methods::TestaddHandle TestaddHandle;
    fields::Test_Field Test_Field;
};
} // namespace skeleton
} // namespace soc_mcu
} // namespace hozon

#endif // HOZON_SOC_MCU_HZSOCMCUSERVERSERVICEINTERFACE_SKELETON_H
