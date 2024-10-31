/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_TSP_PKI_HOZONINTERFACE_TSPPKI_SKELETON_H
#define HOZON_INTERFACE_TSP_PKI_HOZONINTERFACE_TSPPKI_SKELETON_H

#include "ara/com/internal/skeleton/skeleton_adapter.h"
#include "ara/com/internal/skeleton/event_adapter.h"
#include "ara/com/internal/skeleton/field_adapter.h"
#include "ara/com/internal/skeleton/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/interface/tsp_pki/hozoninterface_tsppki_common.h"
#include <cstdint>

namespace hozon {
namespace interface {
namespace tsp_pki {
namespace skeleton {
namespace events
{
}

namespace methods
{
    using RequestHdUuidHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using RequestUploadTokenHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using RequestRemoteConfigHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using ReadPkiStatusHandle = ara::com::internal::skeleton::method::MethodAdapter;
    static constexpr ara::com::internal::EntityId HozonInterface_TspPkiRequestHdUuidId = 60529U; //RequestHdUuid_method_hash
    static constexpr ara::com::internal::EntityId HozonInterface_TspPkiRequestUploadTokenId = 39210U; //RequestUploadToken_method_hash
    static constexpr ara::com::internal::EntityId HozonInterface_TspPkiRequestRemoteConfigId = 27288U; //RequestRemoteConfig_method_hash
    static constexpr ara::com::internal::EntityId HozonInterface_TspPkiReadPkiStatusId = 30124U; //ReadPkiStatus_method_hash
}

namespace fields
{
}

class HozonInterface_TspPkiSkeleton {
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
        const ara::core::Result<void> resultRequestHdUuid = skeletonAdapter->InitializeMethod<ara::core::Future<RequestHdUuidOutput>>(methods::HozonInterface_TspPkiRequestHdUuidId);
        ThrowError(resultRequestHdUuid);
        const ara::core::Result<void> resultRequestUploadToken = skeletonAdapter->InitializeMethod<ara::core::Future<RequestUploadTokenOutput>>(methods::HozonInterface_TspPkiRequestUploadTokenId);
        ThrowError(resultRequestUploadToken);
        const ara::core::Result<void> resultRequestRemoteConfig = skeletonAdapter->InitializeMethod<ara::core::Future<RequestRemoteConfigOutput>>(methods::HozonInterface_TspPkiRequestRemoteConfigId);
        ThrowError(resultRequestRemoteConfig);
        const ara::core::Result<void> resultReadPkiStatus = skeletonAdapter->InitializeMethod<ara::core::Future<ReadPkiStatusOutput>>(methods::HozonInterface_TspPkiReadPkiStatusId);
        ThrowError(resultReadPkiStatus);
    }

    HozonInterface_TspPkiSkeleton& operator=(const HozonInterface_TspPkiSkeleton&) = delete;

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
    using RequestHdUuidOutput = hozon::interface::tsp_pki::methods::RequestHdUuid::Output;
    
    using RequestUploadTokenOutput = hozon::interface::tsp_pki::methods::RequestUploadToken::Output;
    
    using RequestRemoteConfigOutput = hozon::interface::tsp_pki::methods::RequestRemoteConfig::Output;
    
    using ReadPkiStatusOutput = hozon::interface::tsp_pki::methods::ReadPkiStatus::Output;
    
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
    explicit HozonInterface_TspPkiSkeleton(const ara::com::InstanceIdentifier& instanceId,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        : skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::interface::tsp_pki::HozonInterface_TspPki::ServiceIdentifier, instanceId, mode)),
          RequestHdUuidHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_TspPkiRequestHdUuidId),
          RequestUploadTokenHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_TspPkiRequestUploadTokenId),
          RequestRemoteConfigHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_TspPkiRequestRemoteConfigId),
          ReadPkiStatusHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_TspPkiReadPkiStatusId){
        ConstructSkeleton(mode);
    }

    explicit HozonInterface_TspPkiSkeleton(const ara::core::InstanceSpecifier& instanceSpec,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::interface::tsp_pki::HozonInterface_TspPki::ServiceIdentifier, instanceSpec, mode)),
          RequestHdUuidHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_TspPkiRequestHdUuidId),
          RequestUploadTokenHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_TspPkiRequestUploadTokenId),
          RequestRemoteConfigHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_TspPkiRequestRemoteConfigId),
          ReadPkiStatusHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_TspPkiReadPkiStatusId){
        ConstructSkeleton(mode);
    }

    explicit HozonInterface_TspPkiSkeleton(const ara::com::InstanceIdentifierContainer instanceContainer,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::interface::tsp_pki::HozonInterface_TspPki::ServiceIdentifier, instanceContainer, mode)),
          RequestHdUuidHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_TspPkiRequestHdUuidId),
          RequestUploadTokenHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_TspPkiRequestUploadTokenId),
          RequestRemoteConfigHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_TspPkiRequestRemoteConfigId),
          ReadPkiStatusHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_TspPkiReadPkiStatusId){
        ConstructSkeleton(mode);
    }

    HozonInterface_TspPkiSkeleton(const HozonInterface_TspPkiSkeleton&) = delete;

    HozonInterface_TspPkiSkeleton(HozonInterface_TspPkiSkeleton&&) = default;
    HozonInterface_TspPkiSkeleton& operator=(HozonInterface_TspPkiSkeleton&&) = default;
    HozonInterface_TspPkiSkeleton(ConstructionToken&& token) noexcept 
        : skeletonAdapter(token.GetSkeletonAdapter()),
          RequestHdUuidHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_TspPkiRequestHdUuidId),
          RequestUploadTokenHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_TspPkiRequestUploadTokenId),
          RequestRemoteConfigHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_TspPkiRequestRemoteConfigId),
          ReadPkiStatusHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_TspPkiReadPkiStatusId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifier instanceId,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::interface::tsp_pki::HozonInterface_TspPki::ServiceIdentifier, instanceId, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::core::InstanceSpecifier instanceSpec,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::interface::tsp_pki::HozonInterface_TspPki::ServiceIdentifier, instanceSpec, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifierContainer instanceIdContainer,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::interface::tsp_pki::HozonInterface_TspPki::ServiceIdentifier, instanceIdContainer, mode);
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
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<RequestHdUuidOutput>>(methods::HozonInterface_TspPkiRequestHdUuidId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<RequestUploadTokenOutput>>(methods::HozonInterface_TspPkiRequestUploadTokenId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<RequestRemoteConfigOutput>>(methods::HozonInterface_TspPkiRequestRemoteConfigId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<ReadPkiStatusOutput>>(methods::HozonInterface_TspPkiReadPkiStatusId);
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

    virtual ~HozonInterface_TspPkiSkeleton()
    {
        StopOfferService();
    }

    void OfferService()
    {
        skeletonAdapter->RegisterE2EErrorHandler(&HozonInterface_TspPkiSkeleton::E2EErrorHandler, *this);
        skeletonAdapter->RegisterMethod(&HozonInterface_TspPkiSkeleton::RequestHdUuid, *this, methods::HozonInterface_TspPkiRequestHdUuidId);
        skeletonAdapter->RegisterMethod(&HozonInterface_TspPkiSkeleton::RequestUploadToken, *this, methods::HozonInterface_TspPkiRequestUploadTokenId);
        skeletonAdapter->RegisterMethod(&HozonInterface_TspPkiSkeleton::RequestRemoteConfig, *this, methods::HozonInterface_TspPkiRequestRemoteConfigId);
        skeletonAdapter->RegisterMethod(&HozonInterface_TspPkiSkeleton::ReadPkiStatus, *this, methods::HozonInterface_TspPkiReadPkiStatusId);
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
    virtual ara::core::Future<RequestHdUuidOutput> RequestHdUuid() = 0;
    virtual ara::core::Future<RequestUploadTokenOutput> RequestUploadToken() = 0;
    virtual ara::core::Future<RequestRemoteConfigOutput> RequestRemoteConfig() = 0;
    virtual ara::core::Future<ReadPkiStatusOutput> ReadPkiStatus() = 0;
    virtual void E2EErrorHandler(ara::com::e2e::E2EErrorCode, ara::com::e2e::DataID, ara::com::e2e::MessageCounter){}

    methods::RequestHdUuidHandle RequestHdUuidHandle;
    methods::RequestUploadTokenHandle RequestUploadTokenHandle;
    methods::RequestRemoteConfigHandle RequestRemoteConfigHandle;
    methods::ReadPkiStatusHandle ReadPkiStatusHandle;
};
} // namespace skeleton
} // namespace tsp_pki
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_TSP_PKI_HOZONINTERFACE_TSPPKI_SKELETON_H
