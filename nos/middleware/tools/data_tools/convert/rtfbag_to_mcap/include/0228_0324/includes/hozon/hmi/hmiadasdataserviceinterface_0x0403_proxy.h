/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_HMIADASDATASERVICEINTERFACE_0X0403_PROXY_H
#define HOZON_HMI_HMIADASDATASERVICEINTERFACE_0X0403_PROXY_H

#include "ara/com/internal/proxy/proxy_adapter.h"
#include "ara/com/internal/proxy/event_adapter.h"
#include "ara/com/internal/proxy/field_adapter.h"
#include "ara/com/internal/proxy/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/hmi/hmiadasdataserviceinterface_0x0403_common.h"
#include <string>

namespace hozon {
namespace hmi {
namespace proxy {
namespace events {
}

namespace fields {
    using HMIADAS = ara::com::internal::proxy::field::FieldAdapter<::hozon::hmi::ADAS_Dataproperties_Struct>;
    static constexpr ara::com::internal::EntityId HmiADASdataServiceInterface_0x0403HMIADASId = 19957U; //HMIADAS_field_hash
    static constexpr ara::com::internal::EntityId HmiADASdataServiceInterface_0x0403HMIADASGetterId = 57538U; //HMIADAS_getter_hash
}

namespace methods {

} // namespace methods

class HmiADASdataServiceInterface_0x0403Proxy {
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

    virtual ~HmiADASdataServiceInterface_0x0403Proxy()
    {
    }

    explicit HmiADASdataServiceInterface_0x0403Proxy(const vrtf::vcc::api::types::HandleType &handle)
        : proxyAdapter(std::make_unique<ara::com::internal::proxy::ProxyAdapter>(::hozon::hmi::HmiADASdataServiceInterface_0x0403::ServiceIdentifier, handle)),
          HMIADAS(proxyAdapter->GetProxy(), fields::HmiADASdataServiceInterface_0x0403HMIADASId, proxyAdapter->GetHandle(), ::hozon::hmi::HmiADASdataServiceInterface_0x0403::ServiceIdentifier) {
            HMIADAS.SetGetterEntityId(fields::HmiADASdataServiceInterface_0x0403HMIADASGetterId);
            ara::core::Result<void> resultHMIADAS = proxyAdapter->InitializeField<::hozon::hmi::ADAS_Dataproperties_Struct>(HMIADAS);
            ThrowError(resultHMIADAS);
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

    HmiADASdataServiceInterface_0x0403Proxy(const HmiADASdataServiceInterface_0x0403Proxy&) = delete;
    HmiADASdataServiceInterface_0x0403Proxy& operator=(const HmiADASdataServiceInterface_0x0403Proxy&) = delete;

    HmiADASdataServiceInterface_0x0403Proxy(HmiADASdataServiceInterface_0x0403Proxy&&) = default;
    HmiADASdataServiceInterface_0x0403Proxy& operator=(HmiADASdataServiceInterface_0x0403Proxy&&) = default;
    HmiADASdataServiceInterface_0x0403Proxy(ConstructionToken&& token) noexcept
        : proxyAdapter(token.GetProxyAdapter()),
          HMIADAS(proxyAdapter->GetProxy(), fields::HmiADASdataServiceInterface_0x0403HMIADASId, proxyAdapter->GetHandle(), ::hozon::hmi::HmiADASdataServiceInterface_0x0403::ServiceIdentifier) {
        HMIADAS.SetGetterEntityId(fields::HmiADASdataServiceInterface_0x0403HMIADASGetterId);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        const vrtf::vcc::api::types::HandleType &handle)
    {
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> preProxyAdapter =
            std::make_unique<ara::com::internal::proxy::ProxyAdapter>(
               ::hozon::hmi::HmiADASdataServiceInterface_0x0403::ServiceIdentifier, handle);
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            fields::HMIADAS HMIADAS(preProxyAdapter->GetProxy(), fields::HmiADASdataServiceInterface_0x0403HMIADASId, handle, ::hozon::hmi::HmiADASdataServiceInterface_0x0403::ServiceIdentifier);
            HMIADAS.SetGetterEntityId(fields::HmiADASdataServiceInterface_0x0403HMIADASGetterId);
            initResult = preProxyAdapter->InitializeField<::hozon::hmi::ADAS_Dataproperties_Struct>(HMIADAS);
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
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::hozon::hmi::HmiADASdataServiceInterface_0x0403::ServiceIdentifier, instance);
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType> handler,
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::hozon::hmi::HmiADASdataServiceInterface_0x0403::ServiceIdentifier, specifier);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::hozon::hmi::HmiADASdataServiceInterface_0x0403::ServiceIdentifier, instance);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::hozon::hmi::HmiADASdataServiceInterface_0x0403::ServiceIdentifier, specifier);
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
    fields::HMIADAS HMIADAS;
};
} // namespace proxy
} // namespace hmi
} // namespace hozon

#endif // HOZON_HMI_HMIADASDATASERVICEINTERFACE_0X0403_PROXY_H
