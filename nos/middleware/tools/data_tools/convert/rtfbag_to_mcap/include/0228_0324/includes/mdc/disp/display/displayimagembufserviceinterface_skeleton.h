/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_DISP_DISPLAY_DISPLAYIMAGEMBUFSERVICEINTERFACE_SKELETON_H
#define MDC_DISP_DISPLAY_DISPLAYIMAGEMBUFSERVICEINTERFACE_SKELETON_H

#include "ara/com/internal/skeleton/skeleton_adapter.h"
#include "ara/com/internal/skeleton/event_adapter.h"
#include "ara/com/internal/skeleton/field_adapter.h"
#include "ara/com/internal/skeleton/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "mdc/disp/display/displayimagembufserviceinterface_common.h"
#include <cstdint>

namespace mdc {
namespace disp {
namespace display {
namespace skeleton {
namespace events
{
    using DisplayImageMbufEvent = ara::com::internal::skeleton::event::EventAdapter<::ara::display::DisplayImageMbufStruct>;
    static constexpr ara::com::internal::EntityId DisplayImageMbufServiceInterfaceDisplayImageMbufEventId = 56502U; //DisplayImageMbufEvent_event_hash
}

namespace methods
{
}

namespace fields
{
}

class DisplayImageMbufServiceInterfaceSkeleton {
private:
    std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> skeletonAdapter;
    void ConstructSkeleton(const ara::com::MethodCallProcessingMode mode)
    {
        if (mode == ara::com::MethodCallProcessingMode::kEvent) {
            if (!(skeletonAdapter->SetMethodThreadNumber(skeletonAdapter->GetMethodThreadNumber(0U), 1024U))) {
#ifndef NOT_SUPPORT_EXCEPTIONS
                ara::core::ErrorCode errorcode(ara::com::ComErrc::kNetworkBindingFailure);
                throw ara::com::ComException(std::move(errorcode));
#else
                std::cerr << "Error: Not support exception, create skeleton failed!"<< std::endl;
#endif
            }
        }
        const ara::core::Result<void> resultDisplayImageMbufEvent = skeletonAdapter->InitializeEvent(DisplayImageMbufEvent);
        ThrowError(resultDisplayImageMbufEvent);
    }

    DisplayImageMbufServiceInterfaceSkeleton& operator=(const DisplayImageMbufServiceInterfaceSkeleton&) = delete;

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
    };
    explicit DisplayImageMbufServiceInterfaceSkeleton(const ara::com::InstanceIdentifier& instanceId,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        : skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::mdc::disp::display::DisplayImageMbufServiceInterface::ServiceIdentifier, instanceId, mode)),
          DisplayImageMbufEvent(skeletonAdapter->GetSkeleton(), events::DisplayImageMbufServiceInterfaceDisplayImageMbufEventId){
        ConstructSkeleton(mode);
    }

    explicit DisplayImageMbufServiceInterfaceSkeleton(const ara::core::InstanceSpecifier& instanceSpec,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::mdc::disp::display::DisplayImageMbufServiceInterface::ServiceIdentifier, instanceSpec, mode)),
          DisplayImageMbufEvent(skeletonAdapter->GetSkeleton(), events::DisplayImageMbufServiceInterfaceDisplayImageMbufEventId){
        ConstructSkeleton(mode);
    }

    explicit DisplayImageMbufServiceInterfaceSkeleton(const ara::com::InstanceIdentifierContainer instanceContainer,
                           const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        :skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(::mdc::disp::display::DisplayImageMbufServiceInterface::ServiceIdentifier, instanceContainer, mode)),
          DisplayImageMbufEvent(skeletonAdapter->GetSkeleton(), events::DisplayImageMbufServiceInterfaceDisplayImageMbufEventId){
        ConstructSkeleton(mode);
    }

    DisplayImageMbufServiceInterfaceSkeleton(const DisplayImageMbufServiceInterfaceSkeleton&) = delete;

    DisplayImageMbufServiceInterfaceSkeleton(DisplayImageMbufServiceInterfaceSkeleton&&) = default;
    DisplayImageMbufServiceInterfaceSkeleton& operator=(DisplayImageMbufServiceInterfaceSkeleton&&) = default;
    DisplayImageMbufServiceInterfaceSkeleton(ConstructionToken&& token) noexcept 
        : skeletonAdapter(token.GetSkeletonAdapter()),
          DisplayImageMbufEvent(skeletonAdapter->GetSkeleton(), events::DisplayImageMbufServiceInterfaceDisplayImageMbufEventId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifier instanceId,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::mdc::disp::display::DisplayImageMbufServiceInterface::ServiceIdentifier, instanceId, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::core::InstanceSpecifier instanceSpec,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::mdc::disp::display::DisplayImageMbufServiceInterface::ServiceIdentifier, instanceSpec, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        ara::com::InstanceIdentifierContainer instanceIdContainer,
        const ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
    {
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> preSkeletonAdapter =
            std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
                ::mdc::disp::display::DisplayImageMbufServiceInterface::ServiceIdentifier, instanceIdContainer, mode);
        return PreConstructResult(preSkeletonAdapter, mode);
    }

    static ara::core::Result<ConstructionToken> PreConstructResult(
        std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter>& preSkeletonAdapter, const ara::com::MethodCallProcessingMode mode)
    {
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            if (mode == ara::com::MethodCallProcessingMode::kEvent) {
                if(!preSkeletonAdapter->SetMethodThreadNumber(preSkeletonAdapter->GetMethodThreadNumber(0U), 1024U)) {
                    result = false;
                    ara::core::ErrorCode errorcode(ara::com::ComErrc::kNetworkBindingFailure);
                    initResult.EmplaceError(errorcode);
                    break;
                }
            }
            const events::DisplayImageMbufEvent DisplayImageMbufEvent(preSkeletonAdapter->GetSkeleton(), events::DisplayImageMbufServiceInterfaceDisplayImageMbufEventId);
            initResult = preSkeletonAdapter->InitializeEvent(DisplayImageMbufEvent);
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

    virtual ~DisplayImageMbufServiceInterfaceSkeleton()
    {
        StopOfferService();
    }

    void OfferService()
    {
        skeletonAdapter->RegisterE2EErrorHandler(&DisplayImageMbufServiceInterfaceSkeleton::E2EErrorHandler, *this);
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
    virtual void E2EErrorHandler(ara::com::e2e::E2EErrorCode, ara::com::e2e::DataID, ara::com::e2e::MessageCounter){}

    events::DisplayImageMbufEvent DisplayImageMbufEvent;
};
} // namespace skeleton
} // namespace display
} // namespace disp
} // namespace mdc

#endif // MDC_DISP_DISPLAY_DISPLAYIMAGEMBUFSERVICEINTERFACE_SKELETON_H
