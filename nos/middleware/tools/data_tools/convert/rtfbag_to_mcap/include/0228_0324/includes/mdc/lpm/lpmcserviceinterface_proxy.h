/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_LPM_LPMCSERVICEINTERFACE_PROXY_H
#define MDC_LPM_LPMCSERVICEINTERFACE_PROXY_H

#include "ara/com/internal/proxy/proxy_adapter.h"
#include "ara/com/internal/proxy/event_adapter.h"
#include "ara/com/internal/proxy/field_adapter.h"
#include "ara/com/internal/proxy/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "mdc/lpm/lpmcserviceinterface_common.h"
#include <string>

namespace mdc {
namespace lpm {
namespace proxy {
namespace events {
    using MdcWakeupEvent = ara::com::internal::proxy::event::EventAdapter<::mdc::lpm::MdcWakeupEventData>;
    static constexpr ara::com::internal::EntityId LpmcServiceInterfaceMdcWakeupEventId = 53522U; //MdcWakeupEvent_event_hash
}

namespace fields {
}

namespace methods {
static constexpr ara::com::internal::EntityId LpmcServiceInterfaceSetLowPowerModeId = 51001U; //SetLowPowerMode_method_hash
static constexpr ara::com::internal::EntityId LpmcServiceInterfaceSetEnterDeepSleepTimeId = 33809U; //SetEnterDeepSleepTime_method_hash


class SetLowPowerMode {
public:
    using Output = mdc::lpm::methods::SetLowPowerMode::Output;

    SetLowPowerMode(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()()
    {
        return method_();
    }

    ara::com::internal::proxy::method::MethodAdapter<Output> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output> method_;
};

class SetEnterDeepSleepTime {
public:
    using Output = mdc::lpm::methods::SetEnterDeepSleepTime::Output;

    SetEnterDeepSleepTime(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::uint32_t& enterDeepSleepTime)
    {
        return method_(enterDeepSleepTime);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::uint32_t> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::uint32_t> method_;
};
} // namespace methods

class LpmcServiceInterfaceProxy {
private:
    std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> proxyAdapter;
public:
    using HandleType = vrtf::vcc::api::types::HandleType;
    class ConstructionToken {
    public:
        explicit ConstructionToken(std::unique_ptr<ara::com::internal::proxy::ProxyAdapter>& proxy): ptr(std::move(proxy)){}
        explicit ConstructionToken(std::unique_ptr<ara::com::internal::proxy::ProxyAdapter>&& proxy): ptr(std::move(proxy)){}
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
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> GetProxyAdapter()
        {
            return std::move(ptr);
        }
    private:
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> ptr;
    };

    virtual ~LpmcServiceInterfaceProxy()
    {
        MdcWakeupEvent.UnsetReceiveHandler();
        MdcWakeupEvent.Unsubscribe();
    }

    explicit LpmcServiceInterfaceProxy(const vrtf::vcc::api::types::HandleType &handle)
        : proxyAdapter(std::make_unique<ara::com::internal::proxy::ProxyAdapter>(::mdc::lpm::LpmcServiceInterface::ServiceIdentifier, handle)),
          MdcWakeupEvent(proxyAdapter->GetProxy(), events::LpmcServiceInterfaceMdcWakeupEventId, proxyAdapter->GetHandle(), ::mdc::lpm::LpmcServiceInterface::ServiceIdentifier),
          SetLowPowerMode(proxyAdapter->GetProxy(), methods::LpmcServiceInterfaceSetLowPowerModeId),
          SetEnterDeepSleepTime(proxyAdapter->GetProxy(), methods::LpmcServiceInterfaceSetEnterDeepSleepTimeId){
            ara::core::Result<void> resultSetLowPowerMode = proxyAdapter->InitializeMethod<methods::SetLowPowerMode::Output>(methods::LpmcServiceInterfaceSetLowPowerModeId);
            ThrowError(resultSetLowPowerMode);
            ara::core::Result<void> resultSetEnterDeepSleepTime = proxyAdapter->InitializeMethod<methods::SetEnterDeepSleepTime::Output>(methods::LpmcServiceInterfaceSetEnterDeepSleepTimeId);
            ThrowError(resultSetEnterDeepSleepTime);
        }

    void ThrowError(const ara::core::Result<void>& result) const
    {
        if (!(result.HasValue())) {
#ifndef NOT_SUPPORT_EXCEPTIONS
            ara::core::ErrorCode errorcode(result.Error());
            throw ara::com::ComException(std::move(errorcode));
#else
            std::cerr << "Error: Not support exception, create proxy failed!"<< std::endl;
#endif
        }
    }

    LpmcServiceInterfaceProxy(const LpmcServiceInterfaceProxy&) = delete;
    LpmcServiceInterfaceProxy& operator=(const LpmcServiceInterfaceProxy&) = delete;

    LpmcServiceInterfaceProxy(LpmcServiceInterfaceProxy&&) = default;
    LpmcServiceInterfaceProxy& operator=(LpmcServiceInterfaceProxy&&) = default;
    LpmcServiceInterfaceProxy(ConstructionToken&& token) noexcept
        : proxyAdapter(token.GetProxyAdapter()),
          MdcWakeupEvent(proxyAdapter->GetProxy(), events::LpmcServiceInterfaceMdcWakeupEventId, proxyAdapter->GetHandle(), ::mdc::lpm::LpmcServiceInterface::ServiceIdentifier),
          SetLowPowerMode(proxyAdapter->GetProxy(), methods::LpmcServiceInterfaceSetLowPowerModeId),
          SetEnterDeepSleepTime(proxyAdapter->GetProxy(), methods::LpmcServiceInterfaceSetEnterDeepSleepTimeId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        const vrtf::vcc::api::types::HandleType &handle)
    {
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> preProxyAdapter =
            std::make_unique<ara::com::internal::proxy::ProxyAdapter>(
               ::mdc::lpm::LpmcServiceInterface::ServiceIdentifier, handle);
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            const methods::SetLowPowerMode SetLowPowerMode(preProxyAdapter->GetProxy(), methods::LpmcServiceInterfaceSetLowPowerModeId);
            initResult = preProxyAdapter->InitializeMethod<methods::SetLowPowerMode::Output>(methods::LpmcServiceInterfaceSetLowPowerModeId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::SetEnterDeepSleepTime SetEnterDeepSleepTime(preProxyAdapter->GetProxy(), methods::LpmcServiceInterfaceSetEnterDeepSleepTimeId);
            initResult = preProxyAdapter->InitializeMethod<methods::SetEnterDeepSleepTime::Output>(methods::LpmcServiceInterfaceSetEnterDeepSleepTimeId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
        } while(false);

        if (result) {
            ConstructionToken token(std::move(preProxyAdapter));
            return ara::core::Result<ConstructionToken>(std::move(token));
        } else {
            ConstructionToken token(std::move(preProxyAdapter));
            ara::core::Result<ConstructionToken> preResult(std::move(token));
            const ara::core::ErrorCode errorcode(initResult.Error());
            preResult.EmplaceError(errorcode);
            return preResult;
        }
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType>& handler,
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::mdc::lpm::LpmcServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType> handler,
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::mdc::lpm::LpmcServiceInterface::ServiceIdentifier, specifier);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::mdc::lpm::LpmcServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::mdc::lpm::LpmcServiceInterface::ServiceIdentifier, specifier);
    }

    static void StopFindService(const ara::com::FindServiceHandle& handle)
    {
        ara::com::internal::proxy::ProxyAdapter::StopFindService(handle);
    }

    HandleType GetHandle() const
    {
        return proxyAdapter->GetHandle();
    }
    bool SetEventThreadNumber(const std::uint16_t number, const std::uint16_t queueSize)
    {
        return proxyAdapter->SetEventThreadNumber(number, queueSize);
    }
    events::MdcWakeupEvent MdcWakeupEvent;
    methods::SetLowPowerMode SetLowPowerMode;
    methods::SetEnterDeepSleepTime SetEnterDeepSleepTime;
};
} // namespace proxy
} // namespace lpm
} // namespace mdc

#endif // MDC_LPM_LPMCSERVICEINTERFACE_PROXY_H
