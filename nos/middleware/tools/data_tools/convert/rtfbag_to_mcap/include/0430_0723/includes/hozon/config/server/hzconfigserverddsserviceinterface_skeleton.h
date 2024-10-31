/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_CONFIG_SERVER_HZCONFIGSERVERDDSSERVICEINTERFACE_SKELETON_H
#define HOZON_CONFIG_SERVER_HZCONFIGSERVERDDSSERVICEINTERFACE_SKELETON_H

#include "ara/com/internal/skeleton/skeleton_adapter.h"
#include "ara/com/internal/skeleton/event_adapter.h"
#include "ara/com/internal/skeleton/field_adapter.h"
#include "ara/com/internal/skeleton/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/config/server/hzconfigserverddsserviceinterface_common.h"
#include <cstdint>

namespace hozon {
namespace config {
namespace server {
namespace skeleton {
namespace events
{
}

namespace methods
{
    using ReadVehicleConfigHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using WriteVehicleConfigHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using ReadVINConfigHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using WriteVINConfigHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using ReadSNConfigHandle = ara::com::internal::skeleton::method::MethodAdapter;
    using WriteSNConfigHandle = ara::com::internal::skeleton::method::MethodAdapter;
    static constexpr ara::com::internal::EntityId HzConfigServerDdsServiceInterfaceReadVehicleConfigId = 52086U; //ReadVehicleConfig_method_hash
    static constexpr ara::com::internal::EntityId HzConfigServerDdsServiceInterfaceWriteVehicleConfigId = 29793U; //WriteVehicleConfig_method_hash
    static constexpr ara::com::internal::EntityId HzConfigServerDdsServiceInterfaceReadVINConfigId = 27579U; //ReadVINConfig_method_hash
    static constexpr ara::com::internal::EntityId HzConfigServerDdsServiceInterfaceWriteVINConfigId = 44604U; //WriteVINConfig_method_hash
    static constexpr ara::com::internal::EntityId HzConfigServerDdsServiceInterfaceReadSNConfigId = 42610U; //ReadSNConfig_method_hash
    static constexpr ara::com::internal::EntityId HzConfigServerDdsServiceInterfaceWriteSNConfigId = 29526U; //WriteSNConfig_method_hash
}

namespace fields
{
}

class HzConfigServerDdsServiceInterfaceSkeleton {
private:
    std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> skeletonAdapter;
    void ConstructSkeleton(const ara::com::MethodCallProcessingMode mode)
    {
        if (mode == ara::com::MethodCallProcessingMode::kEvent) {
            if (!(skeletonAdapter->SetMethodThreadNumber(skeletonAdapter->GetMethodThreadNumber(6U), 1024U))) {
#ifndef NOT_SUPPORT_EXCEPTIONS
                ara::core::ErrorCode errorcode(ara::com::ComErrc::kNetworkBindingFailure);
                throw ara::com::ComException(std::move(errorcode));
#else
                std::cerr << "Error: Not support exception, create skeleton failed!"<< std::endl;
#endif
            }
        }
        const ara::core::Result<void> resultReadVehicleConfig = skeletonAdapter->InitializeMethod<ara::core::Future<ReadVehicleConfigOutput>>(methods::HzConfigServerDdsServiceInterfaceReadVehicleConfigId);
        ThrowError(resultReadVehicleConfig);
        const ara::core::Result<void> resultWriteVehicleConfig = skeletonAdapter->InitializeMethod<ara::core::Future<WriteVehicleConfigOutput>>(methods::HzConfigServerDdsServiceInterfaceWriteVehicleConfigId);
        ThrowError(resultWriteVehicleConfig);
        const ara::core::Result<void> resultReadVINConfig = skeletonAdapter->InitializeMethod<ara::core::Future<ReadVINConfigOutput>>(methods::HzConfigServerDdsServiceInterfaceReadVINConfigId);
        ThrowError(resultReadVINConfig);
        const ara::core::Result<void> resultWriteVINConfig = skeletonAdapter->InitializeMethod<ara::core::Future<WriteVINConfigOutput>>(methods::HzConfigServerDdsServiceInterfaceWriteVINConfigId);
        ThrowError(resultWriteVINConfig);
        const ara::core::Result<void> resultReadSNConfig = skeletonAdapter->InitializeMethod<ara::core::Future<ReadSNConfigOutput>>(methods::HzConfigServerDdsServiceInterfaceReadSNConfigId);
        ThrowError(resultReadSNConfig);
        const ara::core::Result<void> resultWriteSNConfig = skeletonAdapter->InitializeMethod<ara::core::Future<WriteSNConfigOutput>>(methods::HzConfigServerDdsServiceInterfaceWriteSNConfigId);
        ThrowError(resultWriteSNConfig);
    }

    HzConfigServerDdsServiceInterfaceSkeleton& operator=(const HzConfigServerDdsServiceInterfaceSkeleton&) = delete;

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
    using ReadVehicleConfigOutput = hozon::config::server::methods::ReadVehicleConfig::Output;
    
    using WriteVehicleConfigOutput = hozon::config::server::methods::WriteVehicleConfig::Output;
    
    using ReadVINConfigOutput = hozon::config::server::methods::ReadVINConfig::Output;
    
    using WriteVINConfigOutput = hozon::config::server::methods::WriteVINConfig::Output;
    
    using ReadSNConfigOutput = hozon::config::server::methods::ReadSNConfig::Output;
    
    using WriteSNConfigOutput = hozon::config::server::methods::WriteSNConfig::Output;
    
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
    explicit HzConfigServerDdsServiceInterfaceSkeleton(const ara::com::InstanceIdentifier& instanceId,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        : skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::config::server::HzConfigServerDdsServiceInterface::ServiceIdentifier, instanceId, mode)),
          ReadVehicleConfigHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerDdsServiceInterfaceReadVehicleConfigId),
          WriteVehicleConfigHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerDdsServiceInterfaceWriteVehicleConfigId),
          ReadVINConfigHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerDdsServiceInterfaceReadVINConfigId),
          WriteVINConfigHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerDdsServiceInterfaceWriteVINConfigId),
          ReadSNConfigHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerDdsServiceInterfaceReadSNConfigId),
          WriteSNConfigHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerDdsServiceInterfaceWriteSNConfigId){
        ConstructSkeleton(mode);
    }

    explicit HzConfigServerDdsServiceInterfaceSkeleton(const ara::core::InstanceSpecifier& instanceSpec,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::config::server::HzConfigServerDdsServiceInterface::ServiceIdentifier, instanceSpec, mode)),
          ReadVehicleConfigHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerDdsServiceInterfaceReadVehicleConfigId),
          WriteVehicleConfigHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerDdsServiceInterfaceWriteVehicleConfigId),
          ReadVINConfigHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerDdsServiceInterfaceReadVINConfigId),
          WriteVINConfigHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerDdsServiceInterfaceWriteVINConfigId),
          ReadSNConfigHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerDdsServiceInterfaceReadSNConfigId),
          WriteSNConfigHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerDdsServiceInterfaceWriteSNConfigId){
        ConstructSkeleton(mode);
    }

    explicit HzConfigServerDdsServiceInterfaceSkeleton(const ara::com::InstanceIdentifierContainer instanceContainer,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::hozon::config::server::HzConfigServerDdsServiceInterface::ServiceIdentifier, instanceContainer, mode)),
          ReadVehicleConfigHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerDdsServiceInterfaceReadVehicleConfigId),
          WriteVehicleConfigHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerDdsServiceInterfaceWriteVehicleConfigId),
          ReadVINConfigHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerDdsServiceInterfaceReadVINConfigId),
          WriteVINConfigHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerDdsServiceInterfaceWriteVINConfigId),
          ReadSNConfigHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerDdsServiceInterfaceReadSNConfigId),
          WriteSNConfigHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerDdsServiceInterfaceWriteSNConfigId){
        ConstructSkeleton(mode);
    }

    HzConfigServerDdsServiceInterfaceSkeleton(const HzConfigServerDdsServiceInterfaceSkeleton&) = delete;

    HzConfigServerDdsServiceInterfaceSkeleton(HzConfigServerDdsServiceInterfaceSkeleton&&) = default;
    HzConfigServerDdsServiceInterfaceSkeleton& operator=(HzConfigServerDdsServiceInterfaceSkeleton&&) = default;
    HzConfigServerDdsServiceInterfaceSkeleton(ConstructionToken&& token) noexcept 
        : skeletonAdapter(token.GetSkeletonAdapter()),
          ReadVehicleConfigHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerDdsServiceInterfaceReadVehicleConfigId),
          WriteVehicleConfigHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerDdsServiceInterfaceWriteVehicleConfigId),
          ReadVINConfigHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerDdsServiceInterfaceReadVINConfigId),
          WriteVINConfigHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerDdsServiceInterfaceWriteVINConfigId),
          ReadSNConfigHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerDdsServiceInterfaceReadSNConfigId),
          WriteSNConfigHandle(skeletonAdapter->GetSkeleton(), methods::HzConfigServerDdsServiceInterfaceWriteSNConfigId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifier instanceId,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::config::server::HzConfigServerDdsServiceInterface::ServiceIdentifier, instanceId, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::core::InstanceSpecifier instanceSpec,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::config::server::HzConfigServerDdsServiceInterface::ServiceIdentifier, instanceSpec, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifierContainer instanceIdContainer,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::hozon::config::server::HzConfigServerDdsServiceInterface::ServiceIdentifier, instanceIdContainer, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> PreConstructResult(
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter>& preSkeletonAdapter, const ara::com::MethodCallProcessingMode mode)
    {
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            if (mode == ara::com::MethodCallProcessingMode::kEvent) {
                if(!preSkeletonAdapter->SetMethodThreadNumber(preSkeletonAdapter->GetMethodThreadNumber(6U), 1024U)) {
                    result = false;
                    ara::core::ErrorCode errorcode(ara::com::ComErrc::kNetworkBindingFailure);
                    initResult.EmplaceError(errorcode);
                    break;
                }
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<ReadVehicleConfigOutput>>(methods::HzConfigServerDdsServiceInterfaceReadVehicleConfigId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<WriteVehicleConfigOutput>>(methods::HzConfigServerDdsServiceInterfaceWriteVehicleConfigId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<ReadVINConfigOutput>>(methods::HzConfigServerDdsServiceInterfaceReadVINConfigId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<WriteVINConfigOutput>>(methods::HzConfigServerDdsServiceInterfaceWriteVINConfigId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<ReadSNConfigOutput>>(methods::HzConfigServerDdsServiceInterfaceReadSNConfigId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            initResult = preSkeletonAdapter->InitializeMethod<ara::core::Future<WriteSNConfigOutput>>(methods::HzConfigServerDdsServiceInterfaceWriteSNConfigId);
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

    virtual ~HzConfigServerDdsServiceInterfaceSkeleton()
    {
        StopOfferService();
    }

    void OfferService()
    {
        skeletonAdapter->RegisterE2EErrorHandler(&HzConfigServerDdsServiceInterfaceSkeleton::E2EErrorHandler, *this);
        skeletonAdapter->RegisterMethod(&HzConfigServerDdsServiceInterfaceSkeleton::ReadVehicleConfig, *this, methods::HzConfigServerDdsServiceInterfaceReadVehicleConfigId);
        skeletonAdapter->RegisterMethod(&HzConfigServerDdsServiceInterfaceSkeleton::WriteVehicleConfig, *this, methods::HzConfigServerDdsServiceInterfaceWriteVehicleConfigId);
        skeletonAdapter->RegisterMethod(&HzConfigServerDdsServiceInterfaceSkeleton::ReadVINConfig, *this, methods::HzConfigServerDdsServiceInterfaceReadVINConfigId);
        skeletonAdapter->RegisterMethod(&HzConfigServerDdsServiceInterfaceSkeleton::WriteVINConfig, *this, methods::HzConfigServerDdsServiceInterfaceWriteVINConfigId);
        skeletonAdapter->RegisterMethod(&HzConfigServerDdsServiceInterfaceSkeleton::ReadSNConfig, *this, methods::HzConfigServerDdsServiceInterfaceReadSNConfigId);
        skeletonAdapter->RegisterMethod(&HzConfigServerDdsServiceInterfaceSkeleton::WriteSNConfig, *this, methods::HzConfigServerDdsServiceInterfaceWriteSNConfigId);
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
    virtual ara::core::Future<ReadVehicleConfigOutput> ReadVehicleConfig() = 0;
    virtual ara::core::Future<WriteVehicleConfigOutput> WriteVehicleConfig(const ::hozon::config::server::struct_config_array& vehicleConfigInfo) = 0;
    virtual ara::core::Future<ReadVINConfigOutput> ReadVINConfig() = 0;
    virtual ara::core::Future<WriteVINConfigOutput> WriteVINConfig(const ::String& VIN) = 0;
    virtual ara::core::Future<ReadSNConfigOutput> ReadSNConfig() = 0;
    virtual ara::core::Future<WriteSNConfigOutput> WriteSNConfig(const ::String& SN) = 0;
    virtual void E2EErrorHandler(ara::com::e2e::E2EErrorCode, ara::com::e2e::DataID, ara::com::e2e::MessageCounter){}

    methods::ReadVehicleConfigHandle ReadVehicleConfigHandle;
    methods::WriteVehicleConfigHandle WriteVehicleConfigHandle;
    methods::ReadVINConfigHandle ReadVINConfigHandle;
    methods::WriteVINConfigHandle WriteVINConfigHandle;
    methods::ReadSNConfigHandle ReadSNConfigHandle;
    methods::WriteSNConfigHandle WriteSNConfigHandle;
};
} // namespace skeleton
} // namespace server
} // namespace config
} // namespace hozon

#endif // HOZON_CONFIG_SERVER_HZCONFIGSERVERDDSSERVICEINTERFACE_SKELETON_H
