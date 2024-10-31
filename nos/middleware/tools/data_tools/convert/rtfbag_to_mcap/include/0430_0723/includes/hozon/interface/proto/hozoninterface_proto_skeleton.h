/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_PROTO_HOZONINTERFACE_PROTO_SKELETON_H
#define HOZON_INTERFACE_PROTO_HOZONINTERFACE_PROTO_SKELETON_H

#include "ara/com/internal/skeleton/skeleton_adapter.h"
#include "ara/com/internal/skeleton/event_adapter.h"
#include "ara/com/internal/skeleton/field_adapter.h"
#include "ara/com/internal/skeleton/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/interface/proto/hozoninterface_proto_common.h"
#include <cstdint>

namespace hozon {
namespace interface {
namespace proto {
namespace skeleton {
namespace events
{
}

namespace methods
{
    using ProtoMethod_ReqWithRespHandle = ara::com::internal::skeleton::method::MethodAdapter;
    static constexpr ara::com::internal::EntityId HozonInterface_ProtoProtoMethod_ReqWithRespId = 42085U; //ProtoMethod_ReqWithResp_method_hash
}

namespace fields
{
}

class HozonInterface_ProtoSkeleton {
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
        const ara::core::Result<void> resultProtoMethod_ReqWithResp = skeletonAdapter->InitializeMethod<ara::core::Future<ProtoMethod_ReqWithRespOutput>>(methods::HozonInterface_ProtoProtoMethod_ReqWithRespId);
        ThrowError(resultProtoMethod_ReqWithResp);
    }

    HozonInterface_ProtoSkeleton& operator=(const HozonInterface_ProtoSkeleton&) = delete;

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
    using ProtoMethod_ReqWithRespOutput = hozon::interface::proto::methods::ProtoMethod_ReqWithResp::Output;
    
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
    explicit HozonInterface_ProtoSkeleton(const ara::com::InstanceIdentifier& instanceId,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        : skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::interface::proto::HozonInterface_Proto::ServiceIdentifier, instanceId, mode)),
          ProtoMethod_ReqWithRespHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_ProtoProtoMethod_ReqWithRespId){
        ConstructSkeleton(mode);
    }

    explicit HozonInterface_ProtoSkeleton(const ara::core::InstanceSpecifier& instanceSpec,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::interface::proto::HozonInterface_Proto::ServiceIdentifier, instanceSpec, mode)),
          ProtoMethod_ReqWithRespHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_ProtoProtoMethod_ReqWithRespId){
        ConstructSkeleton(mode);
    }

    explicit HozonInterface_ProtoSkeleton(const ara::com::InstanceIdentifierContainer instanceContainer,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::interface::proto::HozonInterface_Proto::ServiceIdentifier, instanceContainer, mode)),
          ProtoMethod_ReqWithRespHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_ProtoProtoMethod_ReqWithRespId){
        ConstructSkeleton(mode);
    }

    HozonInterface_ProtoSkeleton(const HozonInterface_ProtoSkeleton&) = delete;

    HozonInterface_ProtoSkeleton(HozonInterface_ProtoSkeleton&&) = default;
    HozonInterface_ProtoSkeleton& operator=(HozonInterface_ProtoSkeleton&&) = default;
    HozonInterface_ProtoSkeleton(ConstructionToken&& token) noexcept 
        : skeletonAdapter(token.GetSkeletonAdapter()),
          ProtoMethod_ReqWithRespHandle(skeletonAdapter->GetSkeleton(), methods::HozonInterface_ProtoProtoMethod_ReqWithRespId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifier instanceId,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::interface::proto::HozonInterface_Proto::ServiceIdentifier, instanceId, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::core::InstanceSpecifier instanceSpec,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::interface::proto::HozonInterface_Proto::ServiceIdentifier, instanceSpec, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifierContainer instanceIdContainer,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::interface::proto::HozonInterface_Proto::ServiceIdentifier, instanceIdContainer, mode);
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
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<ProtoMethod_ReqWithRespOutput>>(methods::HozonInterface_ProtoProtoMethod_ReqWithRespId);
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

    virtual ~HozonInterface_ProtoSkeleton()
    {
        StopOfferService();
    }

    void OfferService()
    {
        skeletonAdapter->RegisterE2EErrorHandler(&HozonInterface_ProtoSkeleton::E2EErrorHandler, *this);
        skeletonAdapter->RegisterMethod(&HozonInterface_ProtoSkeleton::ProtoMethod_ReqWithResp, *this, methods::HozonInterface_ProtoProtoMethod_ReqWithRespId);
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
    virtual ara::core::Future<ProtoMethod_ReqWithRespOutput> ProtoMethod_ReqWithResp(const ::String& input) = 0;
    virtual void E2EErrorHandler(ara::com::e2e::E2EErrorCode, ara::com::e2e::DataID, ara::com::e2e::MessageCounter){}

    methods::ProtoMethod_ReqWithRespHandle ProtoMethod_ReqWithRespHandle;
};
} // namespace skeleton
} // namespace proto
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_PROTO_HOZONINTERFACE_PROTO_SKELETON_H
