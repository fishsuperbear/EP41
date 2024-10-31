/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_STATE_MACHINE_HOZONINTERFACE_FIELDTEST_SKELETON_H
#define HOZON_INTERFACE_STATE_MACHINE_HOZONINTERFACE_FIELDTEST_SKELETON_H

#include "ara/com/internal/skeleton/skeleton_adapter.h"
#include "ara/com/internal/skeleton/event_adapter.h"
#include "ara/com/internal/skeleton/field_adapter.h"
#include "ara/com/internal/skeleton/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/interface/state_machine/hozoninterface_fieldtest_common.h"
#include <cstdint>

namespace hozon {
namespace interface {
namespace state_machine {
namespace skeleton {
namespace events
{
}

namespace methods
{
}

namespace fields
{
    using hozonField = ara::com::internal::skeleton::field::FieldAdapter<::hozon::statemachine::StateMachineFrame>;
    static constexpr ara::com::internal::EntityId HozonInterface_FieldTesthozonFieldId = 7370U; //hozonField_field_hash
    static constexpr ara::com::internal::EntityId HozonInterface_FieldTesthozonFieldSetterId = 11353U; //hozonField_setter_hash
    static constexpr ara::com::internal::EntityId HozonInterface_FieldTesthozonFieldGetterId = 1586U; //hozonField_getter_hash
}

class HozonInterface_FieldTestSkeleton {
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
        hozonField.SetSetterEntityId(fields::HozonInterface_FieldTesthozonFieldSetterId);
        hozonField.SetGetterEntityId(fields::HozonInterface_FieldTesthozonFieldGetterId);
        const ara::core::Result<void> resulthozonField = skeletonAdapter->InitializeField<::hozon::statemachine::StateMachineFrame>(hozonField);
        ThrowError(resulthozonField);
    }

    HozonInterface_FieldTestSkeleton& operator=(const HozonInterface_FieldTestSkeleton&) = delete;

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
        fields::hozonField hozonField;
    };
    explicit HozonInterface_FieldTestSkeleton(const ara::com::InstanceIdentifier& instanceId,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        : skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::interface::state_machine::HozonInterface_FieldTest::ServiceIdentifier, instanceId, mode)),
          hozonField(skeletonAdapter->GetSkeleton(), fields::HozonInterface_FieldTesthozonFieldId) {
        ConstructSkeleton(mode);
    }

    explicit HozonInterface_FieldTestSkeleton(const ara::core::InstanceSpecifier& instanceSpec,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::interface::state_machine::HozonInterface_FieldTest::ServiceIdentifier, instanceSpec, mode)),
          hozonField(skeletonAdapter->GetSkeleton(), fields::HozonInterface_FieldTesthozonFieldId) {
        ConstructSkeleton(mode);
    }

    explicit HozonInterface_FieldTestSkeleton(const ara::com::InstanceIdentifierContainer instanceContainer,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::interface::state_machine::HozonInterface_FieldTest::ServiceIdentifier, instanceContainer, mode)),
          hozonField(skeletonAdapter->GetSkeleton(), fields::HozonInterface_FieldTesthozonFieldId) {
        ConstructSkeleton(mode);
    }

    HozonInterface_FieldTestSkeleton(const HozonInterface_FieldTestSkeleton&) = delete;

    HozonInterface_FieldTestSkeleton(HozonInterface_FieldTestSkeleton&&) = default;
    HozonInterface_FieldTestSkeleton& operator=(HozonInterface_FieldTestSkeleton&&) = default;
    HozonInterface_FieldTestSkeleton(ConstructionToken&& token) noexcept 
        : skeletonAdapter(token.GetSkeletonAdapter()),
          hozonField(skeletonAdapter->GetSkeleton(), fields::HozonInterface_FieldTesthozonFieldId) {
        hozonField.SetSetterEntityId(fields::HozonInterface_FieldTesthozonFieldSetterId);
        hozonField.SetGetterEntityId(fields::HozonInterface_FieldTesthozonFieldGetterId);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifier instanceId,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::interface::state_machine::HozonInterface_FieldTest::ServiceIdentifier, instanceId, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::core::InstanceSpecifier instanceSpec,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::interface::state_machine::HozonInterface_FieldTest::ServiceIdentifier, instanceSpec, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifierContainer instanceIdContainer,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::interface::state_machine::HozonInterface_FieldTest::ServiceIdentifier, instanceIdContainer, mode);
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
            fields::hozonField hozonField(preSkeletonAdapter->GetSkeleton(), fields::HozonInterface_FieldTesthozonFieldId);
            hozonField.SetSetterEntityId(fields::HozonInterface_FieldTesthozonFieldSetterId);
            hozonField.SetGetterEntityId(fields::HozonInterface_FieldTesthozonFieldGetterId);
            initResult = preSkeletonAdapter->InitializeField<::hozon::statemachine::StateMachineFrame>(hozonField);
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

    virtual ~HozonInterface_FieldTestSkeleton()
    {
        StopOfferService();
    }

    void OfferService()
    {
        skeletonAdapter->RegisterE2EErrorHandler(&HozonInterface_FieldTestSkeleton::E2EErrorHandler, *this);
        hozonField.VerifyValidity();
        skeletonAdapter->OfferService();
    }
    void StopOfferService()
    {
        skeletonAdapter->StopOfferService();
        hozonField.ResetInitState();
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

    fields::hozonField hozonField;
};
} // namespace skeleton
} // namespace state_machine
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_STATE_MACHINE_HOZONINTERFACE_FIELDTEST_SKELETON_H
