/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Generated by VRTF CM-Generator
 */

#ifndef RTF_MAINTAIND_RTFMAINTAINDAPPSERVICE_SKELETON_H
#define RTF_MAINTAIND_RTFMAINTAINDAPPSERVICE_SKELETON_H

#include "ara/com/internal/skeleton/skeleton_adapter.h"
#include "ara/com/internal/skeleton/event_adapter.h"
#include "ara/com/internal/skeleton/field_adapter.h"
#include "ara/com/internal/skeleton/method_adapter.h"
#include "rtf/maintaind/rtfmaintaindappservice_common.h"
#include <cstdint>

namespace rtf {
namespace maintaind {
namespace skeleton {
namespace events {
    using SwitchApplicationLatency = ara::com::internal::skeleton::event::EventAdapter<::rtf::maintaind::LatencySwitch>;
    // SwitchApplicationLatency_event_hash
    static constexpr ara::com::internal::EntityId SwitchApplicationLatencyId = 30161U;
}

namespace methods {
    using RegisterAppInfoHandle = ara::com::internal::skeleton::method::MethodAdapter;
    static constexpr ara::com::internal::EntityId RegisterAppInfoId = 60621U;
}

namespace fields {
}

class RTFMaintaindAppServiceSkeleton {
private:
    std::unique_ptr<ara::com::internal::skeleton::SkeletonAdapter> skeletonAdapter;
    void ConstructSkeleton(ara::com::MethodCallProcessingMode mode)
    {
        bool result = true;
        if (mode == ara::com::MethodCallProcessingMode::kEvent) {
            std::uint16_t threadNum = 8U;
            std::uint16_t taskNum = 512U;
            result = result && skeletonAdapter->SetMethodThreadNumber(
                skeletonAdapter->GetMethodThreadNumber(threadNum), taskNum);
        }
        result = result && ((skeletonAdapter->InitializeEvent(SwitchApplicationLatency)).HasValue());
        result = result && ((skeletonAdapter->InitializeMethod<ara::core::Future<RegisterAppInfoOutput>>(
            methods::RegisterAppInfoId)).HasValue());
        if (result == false) {
#ifndef NOT_SUPPORT_EXCEPTIONS
            ara::core::ErrorCode errorcode(ara::com::ComErrc::kNetworkBindingFailure);
            throw ara::com::ComException(std::move(errorcode));
#else
            std::cout << "Error: Not support exception, create skeleton failed!"<< std::endl;
#endif
        }
    }
public:
    struct RegisterAppInfoOutput {
        ::rtf::maintaind::ReturnCode result;

        static bool IsPlane() noexcept
        {
            return false;
        }

        using IsEnumerableTag = void;
        template<typename F>
        void enumerate(F& fun) noexcept
        {
            fun(result);
        }

        template<typename F>
        void enumerate(F& fun) const noexcept
        {
            fun(result);
        }

        bool operator == (const RegisterAppInfoOutput& t) const noexcept
        {
            return (result == t.result);
        }
    };

    struct RegisterNodePidInfoOutput {
        ::rtf::maintaind::ReturnCode result;

        static bool IsPlane() noexcept
        {
            return false;
        }

        using IsEnumerableTag = void;
        template<typename F>
        void enumerate(F& fun) noexcept
        {
            fun(result);
        }

        template<typename F>
        void enumerate(F& fun) const noexcept
        {
            fun(result);
        }

        bool operator == (const RegisterNodePidInfoOutput& t) const noexcept
        {
            return (result == t.result);
        }
    };

    struct UnregisterMethodInfoOutput {
        ::rtf::maintaind::ReturnCode result;

        static bool IsPlane() noexcept
        {
            return false;
        }

        using IsEnumerableTag = void;
        template<typename F>
        void enumerate(F& fun) noexcept
        {
            fun(result);
        }

        template<typename F>
        void enumerate(F& fun) const noexcept
        {
            fun(result);
        }

        bool operator == (const UnregisterMethodInfoOutput& t) const noexcept
        {
            return (result == t.result);
        }
    };

    explicit RTFMaintaindAppServiceSkeleton(ara::com::InstanceIdentifier instanceId,
                           ara::com::MethodCallProcessingMode mode = ara::com::MethodCallProcessingMode::kEvent)
        : skeletonAdapter(std::make_unique<ara::com::internal::skeleton::SkeletonAdapter>(
         ::rtf::maintaind::RTFMaintaindAppService::ServiceIdentifier, instanceId, mode)),
          SwitchApplicationLatency(skeletonAdapter->GetSkeleton(), events::SwitchApplicationLatencyId),
          RegisterAppInfoHandle(skeletonAdapter->GetSkeleton(), methods::RegisterAppInfoId) {
        ConstructSkeleton(mode);
    }

    RTFMaintaindAppServiceSkeleton(const RTFMaintaindAppServiceSkeleton&) = delete;
    RTFMaintaindAppServiceSkeleton& operator=(const RTFMaintaindAppServiceSkeleton&) = delete;

    RTFMaintaindAppServiceSkeleton(RTFMaintaindAppServiceSkeleton&& other) = default;
    RTFMaintaindAppServiceSkeleton& operator=(RTFMaintaindAppServiceSkeleton&& other) = default;
    virtual ~RTFMaintaindAppServiceSkeleton()
    {
        StopOfferService();
    }

    void OfferService()
    {
        skeletonAdapter->RegisterMethod(&RTFMaintaindAppServiceSkeleton::RegisterAppInfo,
            *this, methods::RegisterAppInfoId);
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

    virtual ara::core::Future<RegisterAppInfoOutput> RegisterAppInfo(
        const ::rtf::maintaind::AppRegisterInfo& appInfo) = 0;

    events::SwitchApplicationLatency SwitchApplicationLatency;
    methods::RegisterAppInfoHandle RegisterAppInfoHandle;
};
} // namespace skeleton
} // namespace maintaind
} // namespace rtf

#endif // RTF_MAINTAIND_RTFMAINTAINDAPPSERVICE_SKELETON_H