/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_LATENCY_TEST_1K_HOZONINTERFACE_LATENCYTEST1K_PROXY_H
#define HOZON_INTERFACE_LATENCY_TEST_1K_HOZONINTERFACE_LATENCYTEST1K_PROXY_H

#include "ara/com/internal/proxy/proxy_adapter.h"
#include "ara/com/internal/proxy/event_adapter.h"
#include "ara/com/internal/proxy/field_adapter.h"
#include "ara/com/internal/proxy/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/interface/latency_test_1k/hozoninterface_latencytest1k_common.h"
#include <string>

namespace hozon {
namespace interface {
namespace latency_test_1k {
namespace proxy {
namespace events {
    using hozonEvent = ara::com::internal::proxy::event::EventAdapter<::StandardArray1KB>;
    static constexpr ara::com::internal::EntityId HozonInterface_LatencyTest1KhozonEventId = 32222U; //hozonEvent_event_hash
}

namespace fields {
}

namespace methods {

} // namespace methods

class HozonInterface_LatencyTest1KProxy {
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

    virtual ~HozonInterface_LatencyTest1KProxy()
    {
        hozonEvent.UnsetReceiveHandler();
        hozonEvent.Unsubscribe();
    }

    explicit HozonInterface_LatencyTest1KProxy(const vrtf::vcc::api::types::HandleType &handle)
        : proxyAdapter(std::make_unique<ara::com::internal::proxy::ProxyAdapter>(::hozon::interface::latency_test_1k::HozonInterface_LatencyTest1K::ServiceIdentifier, handle)),
          hozonEvent(proxyAdapter->GetProxy(), events::HozonInterface_LatencyTest1KhozonEventId, proxyAdapter->GetHandle(), ::hozon::interface::latency_test_1k::HozonInterface_LatencyTest1K::ServiceIdentifier){
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

    HozonInterface_LatencyTest1KProxy(const HozonInterface_LatencyTest1KProxy&) = delete;
    HozonInterface_LatencyTest1KProxy& operator=(const HozonInterface_LatencyTest1KProxy&) = delete;

    HozonInterface_LatencyTest1KProxy(HozonInterface_LatencyTest1KProxy&&) = default;
    HozonInterface_LatencyTest1KProxy& operator=(HozonInterface_LatencyTest1KProxy&&) = default;
    HozonInterface_LatencyTest1KProxy(ConstructionToken&& token) noexcept
        : proxyAdapter(token.GetProxyAdapter()),
          hozonEvent(proxyAdapter->GetProxy(), events::HozonInterface_LatencyTest1KhozonEventId, proxyAdapter->GetHandle(), ::hozon::interface::latency_test_1k::HozonInterface_LatencyTest1K::ServiceIdentifier){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        const vrtf::vcc::api::types::HandleType &handle)
    {
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> preProxyAdapter =
            std::make_unique<ara::com::internal::proxy::ProxyAdapter>(
               ::hozon::interface::latency_test_1k::HozonInterface_LatencyTest1K::ServiceIdentifier, handle);
        const bool result = true;
        const ara::core::Result<void> initResult;
        do {
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
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::hozon::interface::latency_test_1k::HozonInterface_LatencyTest1K::ServiceIdentifier, instance);
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType> handler,
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::hozon::interface::latency_test_1k::HozonInterface_LatencyTest1K::ServiceIdentifier, specifier);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::hozon::interface::latency_test_1k::HozonInterface_LatencyTest1K::ServiceIdentifier, instance);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::hozon::interface::latency_test_1k::HozonInterface_LatencyTest1K::ServiceIdentifier, specifier);
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
    events::hozonEvent hozonEvent;
};
} // namespace proxy
} // namespace latency_test_1k
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_LATENCY_TEST_1K_HOZONINTERFACE_LATENCYTEST1K_PROXY_H
