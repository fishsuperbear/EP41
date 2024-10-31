/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_HMIAPADASDATASERVICEINTERFACE_0X0404_SKELETON_H
#define HOZON_HMI_HMIAPADASDATASERVICEINTERFACE_0X0404_SKELETON_H

#include "ara/com/internal/skeleton/skeleton_adapter.h"
#include "ara/com/internal/skeleton/event_adapter.h"
#include "ara/com/internal/skeleton/field_adapter.h"
#include "ara/com/internal/skeleton/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/hmi/hmiapadasdataserviceinterface_0x0404_common.h"
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
    using HMIAPA = ara::com::internal::skeleton::field::FieldAdapter<::hozon::hmi::APA_Dataproperties_Struct>;
    using HMIHPP = ara::com::internal::skeleton::field::FieldAdapter<::hozon::hmi::HPP_Path_Struct>;
    using HMINNS = ara::com::internal::skeleton::field::FieldAdapter<::hozon::hmi::NNS_Info_Struct>;
    using HMIINS = ara::com::internal::skeleton::field::FieldAdapter<::hozon::hmi::Ins_Info_Struct>;
    static constexpr ara::com::internal::EntityId HmiAPADASdataServiceInterface_0x0404HMIAPAId = 4027U; //HMIAPA_field_hash
    static constexpr ara::com::internal::EntityId HmiAPADASdataServiceInterface_0x0404HMIAPAGetterId = 23624U; //HMIAPA_getter_hash
    static constexpr ara::com::internal::EntityId HmiAPADASdataServiceInterface_0x0404HMIHPPId = 64490U; //HMIHPP_field_hash
    static constexpr ara::com::internal::EntityId HmiAPADASdataServiceInterface_0x0404HMIHPPGetterId = 3607U; //HMIHPP_getter_hash
    static constexpr ara::com::internal::EntityId HmiAPADASdataServiceInterface_0x0404HMINNSId = 2757U; //HMINNS_field_hash
    static constexpr ara::com::internal::EntityId HmiAPADASdataServiceInterface_0x0404HMINNSSetterId = 63474U; //HMINNS_setter_hash
    static constexpr ara::com::internal::EntityId HmiAPADASdataServiceInterface_0x0404HMIINSId = 38289U; //HMIINS_field_hash
    static constexpr ara::com::internal::EntityId HmiAPADASdataServiceInterface_0x0404HMIINSSetterId = 57256U; //HMIINS_setter_hash
}

class HmiAPADASdataServiceInterface_0x0404Skeleton {
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
        HMIAPA.SetGetterEntityId(fields::HmiAPADASdataServiceInterface_0x0404HMIAPAGetterId);
        const ara::core::Result<void> resultHMIAPA = skeletonAdapter->InitializeField<::hozon::hmi::APA_Dataproperties_Struct>(HMIAPA);
        ThrowError(resultHMIAPA);
        HMIHPP.SetGetterEntityId(fields::HmiAPADASdataServiceInterface_0x0404HMIHPPGetterId);
        const ara::core::Result<void> resultHMIHPP = skeletonAdapter->InitializeField<::hozon::hmi::HPP_Path_Struct>(HMIHPP);
        ThrowError(resultHMIHPP);
        HMINNS.SetSetterEntityId(fields::HmiAPADASdataServiceInterface_0x0404HMINNSSetterId);
        const ara::core::Result<void> resultHMINNS = skeletonAdapter->InitializeField<::hozon::hmi::NNS_Info_Struct>(HMINNS);
        ThrowError(resultHMINNS);
        HMIINS.SetSetterEntityId(fields::HmiAPADASdataServiceInterface_0x0404HMIINSSetterId);
        const ara::core::Result<void> resultHMIINS = skeletonAdapter->InitializeField<::hozon::hmi::Ins_Info_Struct>(HMIINS);
        ThrowError(resultHMIINS);
    }

    HmiAPADASdataServiceInterface_0x0404Skeleton& operator=(const HmiAPADASdataServiceInterface_0x0404Skeleton&) = delete;

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
        fields::HMIAPA HMIAPA;
        fields::HMIHPP HMIHPP;
        fields::HMINNS HMINNS;
        fields::HMIINS HMIINS;
    };
    explicit HmiAPADASdataServiceInterface_0x0404Skeleton(const ara::com::InstanceIdentifier& instanceId,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        : skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::hmi::HmiAPADASdataServiceInterface_0x0404::ServiceIdentifier, instanceId, mode)),
          HMIAPA(skeletonAdapter->GetSkeleton(), fields::HmiAPADASdataServiceInterface_0x0404HMIAPAId),
          HMIHPP(skeletonAdapter->GetSkeleton(), fields::HmiAPADASdataServiceInterface_0x0404HMIHPPId),
          HMINNS(skeletonAdapter->GetSkeleton(), fields::HmiAPADASdataServiceInterface_0x0404HMINNSId),
          HMIINS(skeletonAdapter->GetSkeleton(), fields::HmiAPADASdataServiceInterface_0x0404HMIINSId) {
        ConstructSkeleton(mode);
    }

    explicit HmiAPADASdataServiceInterface_0x0404Skeleton(const ara::core::InstanceSpecifier& instanceSpec,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::hmi::HmiAPADASdataServiceInterface_0x0404::ServiceIdentifier, instanceSpec, mode)),
          HMIAPA(skeletonAdapter->GetSkeleton(), fields::HmiAPADASdataServiceInterface_0x0404HMIAPAId),
          HMIHPP(skeletonAdapter->GetSkeleton(), fields::HmiAPADASdataServiceInterface_0x0404HMIHPPId),
          HMINNS(skeletonAdapter->GetSkeleton(), fields::HmiAPADASdataServiceInterface_0x0404HMINNSId),
          HMIINS(skeletonAdapter->GetSkeleton(), fields::HmiAPADASdataServiceInterface_0x0404HMIINSId) {
        ConstructSkeleton(mode);
    }

    explicit HmiAPADASdataServiceInterface_0x0404Skeleton(const ara::com::InstanceIdentifierContainer instanceContainer,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::hmi::HmiAPADASdataServiceInterface_0x0404::ServiceIdentifier, instanceContainer, mode)),
          HMIAPA(skeletonAdapter->GetSkeleton(), fields::HmiAPADASdataServiceInterface_0x0404HMIAPAId),
          HMIHPP(skeletonAdapter->GetSkeleton(), fields::HmiAPADASdataServiceInterface_0x0404HMIHPPId),
          HMINNS(skeletonAdapter->GetSkeleton(), fields::HmiAPADASdataServiceInterface_0x0404HMINNSId),
          HMIINS(skeletonAdapter->GetSkeleton(), fields::HmiAPADASdataServiceInterface_0x0404HMIINSId) {
        ConstructSkeleton(mode);
    }

    HmiAPADASdataServiceInterface_0x0404Skeleton(const HmiAPADASdataServiceInterface_0x0404Skeleton&) = delete;

    HmiAPADASdataServiceInterface_0x0404Skeleton(HmiAPADASdataServiceInterface_0x0404Skeleton&&) = default;
    HmiAPADASdataServiceInterface_0x0404Skeleton& operator=(HmiAPADASdataServiceInterface_0x0404Skeleton&&) = default;
    HmiAPADASdataServiceInterface_0x0404Skeleton(ConstructionToken&& token) noexcept 
        : skeletonAdapter(token.GetSkeletonAdapter()),
          HMIAPA(skeletonAdapter->GetSkeleton(), fields::HmiAPADASdataServiceInterface_0x0404HMIAPAId),
          HMIHPP(skeletonAdapter->GetSkeleton(), fields::HmiAPADASdataServiceInterface_0x0404HMIHPPId),
          HMINNS(skeletonAdapter->GetSkeleton(), fields::HmiAPADASdataServiceInterface_0x0404HMINNSId),
          HMIINS(skeletonAdapter->GetSkeleton(), fields::HmiAPADASdataServiceInterface_0x0404HMIINSId) {
        HMIAPA.SetGetterEntityId(fields::HmiAPADASdataServiceInterface_0x0404HMIAPAGetterId);
        HMIHPP.SetGetterEntityId(fields::HmiAPADASdataServiceInterface_0x0404HMIHPPGetterId);
        HMINNS.SetSetterEntityId(fields::HmiAPADASdataServiceInterface_0x0404HMINNSSetterId);
        HMIINS.SetSetterEntityId(fields::HmiAPADASdataServiceInterface_0x0404HMIINSSetterId);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifier instanceId,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::hmi::HmiAPADASdataServiceInterface_0x0404::ServiceIdentifier, instanceId, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::core::InstanceSpecifier instanceSpec,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::hmi::HmiAPADASdataServiceInterface_0x0404::ServiceIdentifier, instanceSpec, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifierContainer instanceIdContainer,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::hmi::HmiAPADASdataServiceInterface_0x0404::ServiceIdentifier, instanceIdContainer, mode);
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
            fields::HMIAPA HMIAPA(preSkeletonAdapter->GetSkeleton(), fields::HmiAPADASdataServiceInterface_0x0404HMIAPAId);
            HMIAPA.SetGetterEntityId(fields::HmiAPADASdataServiceInterface_0x0404HMIAPAGetterId);
            initResult = preSkeletonAdapter->InitializeField<::hozon::hmi::APA_Dataproperties_Struct>(HMIAPA);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            fields::HMIHPP HMIHPP(preSkeletonAdapter->GetSkeleton(), fields::HmiAPADASdataServiceInterface_0x0404HMIHPPId);
            HMIHPP.SetGetterEntityId(fields::HmiAPADASdataServiceInterface_0x0404HMIHPPGetterId);
            initResult = preSkeletonAdapter->InitializeField<::hozon::hmi::HPP_Path_Struct>(HMIHPP);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            fields::HMINNS HMINNS(preSkeletonAdapter->GetSkeleton(), fields::HmiAPADASdataServiceInterface_0x0404HMINNSId);
            HMINNS.SetSetterEntityId(fields::HmiAPADASdataServiceInterface_0x0404HMINNSSetterId);
            initResult = preSkeletonAdapter->InitializeField<::hozon::hmi::NNS_Info_Struct>(HMINNS);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            fields::HMIINS HMIINS(preSkeletonAdapter->GetSkeleton(), fields::HmiAPADASdataServiceInterface_0x0404HMIINSId);
            HMIINS.SetSetterEntityId(fields::HmiAPADASdataServiceInterface_0x0404HMIINSSetterId);
            initResult = preSkeletonAdapter->InitializeField<::hozon::hmi::Ins_Info_Struct>(HMIINS);
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

    virtual ~HmiAPADASdataServiceInterface_0x0404Skeleton()
    {
        StopOfferService();
    }

    void OfferService()
    {
        skeletonAdapter->RegisterE2EErrorHandler(&HmiAPADASdataServiceInterface_0x0404Skeleton::E2EErrorHandler, *this);
        HMIAPA.VerifyValidity();
        HMIHPP.VerifyValidity();
        HMINNS.VerifyValidity();
        HMIINS.VerifyValidity();
        skeletonAdapter->OfferService();
    }
    void StopOfferService()
    {
        skeletonAdapter->StopOfferService();
        HMIAPA.ResetInitState();
        HMIHPP.ResetInitState();
        HMINNS.ResetInitState();
        HMIINS.ResetInitState();
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

    fields::HMIAPA HMIAPA;
    fields::HMIHPP HMIHPP;
    fields::HMINNS HMINNS;
    fields::HMIINS HMIINS;
};
} // namespace skeleton
} // namespace hmi
} // namespace hozon

#endif // HOZON_HMI_HMIAPADASDATASERVICEINTERFACE_0X0404_SKELETON_H
