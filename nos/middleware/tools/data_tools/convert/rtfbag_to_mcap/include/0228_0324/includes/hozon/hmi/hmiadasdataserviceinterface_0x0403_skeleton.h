/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_HMIADASDATASERVICEINTERFACE_0X0403_SKELETON_H
#define HOZON_HMI_HMIADASDATASERVICEINTERFACE_0X0403_SKELETON_H

#include "ara/com/internal/skeleton/skeleton_adapter.h"
#include "ara/com/internal/skeleton/event_adapter.h"
#include "ara/com/internal/skeleton/field_adapter.h"
#include "ara/com/internal/skeleton/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/hmi/hmiadasdataserviceinterface_0x0403_common.h"
#include <cstdint>

namespace hozon {
namespace hmi {
namespace skeleton {
namespace events
{
}

namespace methods
{
}

namespace fields
{
    using HMIADAS = ara::com::internal::skeleton::field::FieldAdapter<::hozon::hmi::ADAS_Dataproperties_Struct>;
    static constexpr ara::com::internal::EntityId HmiADASdataServiceInterface_0x0403HMIADASId = 19957U; //HMIADAS_field_hash
    static constexpr ara::com::internal::EntityId HmiADASdataServiceInterface_0x0403HMIADASGetterId = 57538U; //HMIADAS_getter_hash
}

class HmiADASdataServiceInterface_0x0403Skeleton {
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
        HMIADAS.SetGetterEntityId(fields::HmiADASdataServiceInterface_0x0403HMIADASGetterId);
        const ara::core::Result<void> resultHMIADAS = skeletonAdapter->InitializeField<::hozon::hmi::ADAS_Dataproperties_Struct>(HMIADAS);
        ThrowError(resultHMIADAS);
    }

    HmiADASdataServiceInterface_0x0403Skeleton& operator=(const HmiADASdataServiceInterface_0x0403Skeleton&) = delete;

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
        fields::HMIADAS HMIADAS;
    };
    explicit HmiADASdataServiceInterface_0x0403Skeleton(const ara::com::InstanceIdentifier& instanceId,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        : skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::hmi::HmiADASdataServiceInterface_0x0403::ServiceIdentifier, instanceId, mode)),
          HMIADAS(skeletonAdapter->GetSkeleton(), fields::HmiADASdataServiceInterface_0x0403HMIADASId) {
        ConstructSkeleton(mode);
    }

    explicit HmiADASdataServiceInterface_0x0403Skeleton(const ara::core::InstanceSpecifier& instanceSpec,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::hmi::HmiADASdataServiceInterface_0x0403::ServiceIdentifier, instanceSpec, mode)),
          HMIADAS(skeletonAdapter->GetSkeleton(), fields::HmiADASdataServiceInterface_0x0403HMIADASId) {
        ConstructSkeleton(mode);
    }

    explicit HmiADASdataServiceInterface_0x0403Skeleton(const ara::com::InstanceIdentifierContainer instanceContainer,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::hmi::HmiADASdataServiceInterface_0x0403::ServiceIdentifier, instanceContainer, mode)),
          HMIADAS(skeletonAdapter->GetSkeleton(), fields::HmiADASdataServiceInterface_0x0403HMIADASId) {
        ConstructSkeleton(mode);
    }

    HmiADASdataServiceInterface_0x0403Skeleton(const HmiADASdataServiceInterface_0x0403Skeleton&) = delete;

    HmiADASdataServiceInterface_0x0403Skeleton(HmiADASdataServiceInterface_0x0403Skeleton&&) = default;
    HmiADASdataServiceInterface_0x0403Skeleton& operator=(HmiADASdataServiceInterface_0x0403Skeleton&&) = default;
    HmiADASdataServiceInterface_0x0403Skeleton(ConstructionToken&& token) noexcept 
        : skeletonAdapter(token.GetSkeletonAdapter()),
          HMIADAS(skeletonAdapter->GetSkeleton(), fields::HmiADASdataServiceInterface_0x0403HMIADASId) {
        HMIADAS.SetGetterEntityId(fields::HmiADASdataServiceInterface_0x0403HMIADASGetterId);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifier instanceId,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::hmi::HmiADASdataServiceInterface_0x0403::ServiceIdentifier, instanceId, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::core::InstanceSpecifier instanceSpec,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::hmi::HmiADASdataServiceInterface_0x0403::ServiceIdentifier, instanceSpec, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifierContainer instanceIdContainer,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::hmi::HmiADASdataServiceInterface_0x0403::ServiceIdentifier, instanceIdContainer, mode);
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
            fields::HMIADAS HMIADAS(preSkeletonAdapter->GetSkeleton(), fields::HmiADASdataServiceInterface_0x0403HMIADASId);
            HMIADAS.SetGetterEntityId(fields::HmiADASdataServiceInterface_0x0403HMIADASGetterId);
            initResult = preSkeletonAdapter->InitializeField<::hozon::hmi::ADAS_Dataproperties_Struct>(HMIADAS);
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

    virtual ~HmiADASdataServiceInterface_0x0403Skeleton()
    {
        StopOfferService();
    }

    void OfferService()
    {
        skeletonAdapter->RegisterE2EErrorHandler(&HmiADASdataServiceInterface_0x0403Skeleton::E2EErrorHandler, *this);
        HMIADAS.VerifyValidity();
        skeletonAdapter->OfferService();
    }
    void StopOfferService()
    {
        skeletonAdapter->StopOfferService();
        HMIADAS.ResetInitState();
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

    fields::HMIADAS HMIADAS;
};
} // namespace skeleton
} // namespace hmi
} // namespace hozon

#endif // HOZON_HMI_HMIADASDATASERVICEINTERFACE_0X0403_SKELETON_H
