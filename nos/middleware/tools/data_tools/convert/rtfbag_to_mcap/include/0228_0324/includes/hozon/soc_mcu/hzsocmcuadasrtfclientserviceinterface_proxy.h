/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_HZSOCMCUADASRTFCLIENTSERVICEINTERFACE_PROXY_H
#define HOZON_SOC_MCU_HZSOCMCUADASRTFCLIENTSERVICEINTERFACE_PROXY_H

#include "ara/com/internal/proxy/proxy_adapter.h"
#include "ara/com/internal/proxy/event_adapter.h"
#include "ara/com/internal/proxy/field_adapter.h"
#include "ara/com/internal/proxy/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/soc_mcu/hzsocmcuadasrtfclientserviceinterface_common.h"
#include <string>

namespace hozon {
namespace soc_mcu {
namespace proxy {
namespace events {
    using McuADASRtfEvent = ara::com::internal::proxy::event::EventAdapter<::hozon::soc2mcu::ADAS_DataRecordFrame_soc>;
    static constexpr ara::com::internal::EntityId HzSocMcuAdasRtfClientServiceInterfaceMcuADASRtfEventId = 50653U; //McuADASRtfEvent_event_hash
}

namespace fields {
}

namespace methods {

} // namespace methods

class HzSocMcuAdasRtfClientServiceInterfaceProxy {
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

    virtual ~HzSocMcuAdasRtfClientServiceInterfaceProxy()
    {
        McuADASRtfEvent.UnsetReceiveHandler();
        McuADASRtfEvent.Unsubscribe();
    }

    explicit HzSocMcuAdasRtfClientServiceInterfaceProxy(const vrtf::vcc::api::types::HandleType &handle)
        : proxyAdapter(std::make_unique<ara::com::internal::proxy::ProxyAdapter>(::hozon::soc_mcu::HzSocMcuAdasRtfClientServiceInterface::ServiceIdentifier, handle)),
          McuADASRtfEvent(proxyAdapter->GetProxy(), events::HzSocMcuAdasRtfClientServiceInterfaceMcuADASRtfEventId, proxyAdapter->GetHandle(), ::hozon::soc_mcu::HzSocMcuAdasRtfClientServiceInterface::ServiceIdentifier){
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

    HzSocMcuAdasRtfClientServiceInterfaceProxy(const HzSocMcuAdasRtfClientServiceInterfaceProxy&) = delete;
    HzSocMcuAdasRtfClientServiceInterfaceProxy& operator=(const HzSocMcuAdasRtfClientServiceInterfaceProxy&) = delete;

    HzSocMcuAdasRtfClientServiceInterfaceProxy(HzSocMcuAdasRtfClientServiceInterfaceProxy&&) = default;
    HzSocMcuAdasRtfClientServiceInterfaceProxy& operator=(HzSocMcuAdasRtfClientServiceInterfaceProxy&&) = default;
    HzSocMcuAdasRtfClientServiceInterfaceProxy(ConstructionToken&& token) noexcept
        : proxyAdapter(token.GetProxyAdapter()),
          McuADASRtfEvent(proxyAdapter->GetProxy(), events::HzSocMcuAdasRtfClientServiceInterfaceMcuADASRtfEventId, proxyAdapter->GetHandle(), ::hozon::soc_mcu::HzSocMcuAdasRtfClientServiceInterface::ServiceIdentifier){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        const vrtf::vcc::api::types::HandleType &handle)
    {
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> preProxyAdapter =
            std::make_unique<ara::com::internal::proxy::ProxyAdapter>(
               ::hozon::soc_mcu::HzSocMcuAdasRtfClientServiceInterface::ServiceIdentifier, handle);
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
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::hozon::soc_mcu::HzSocMcuAdasRtfClientServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType> handler,
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::hozon::soc_mcu::HzSocMcuAdasRtfClientServiceInterface::ServiceIdentifier, specifier);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::hozon::soc_mcu::HzSocMcuAdasRtfClientServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::hozon::soc_mcu::HzSocMcuAdasRtfClientServiceInterface::ServiceIdentifier, specifier);
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
    events::McuADASRtfEvent McuADASRtfEvent;
};
} // namespace proxy
} // namespace soc_mcu
} // namespace hozon

#endif // HOZON_SOC_MCU_HZSOCMCUADASRTFCLIENTSERVICEINTERFACE_PROXY_H
