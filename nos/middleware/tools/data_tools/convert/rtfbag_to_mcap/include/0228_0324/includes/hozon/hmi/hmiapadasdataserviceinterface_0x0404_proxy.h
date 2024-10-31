/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_HMI_HMIAPADASDATASERVICEINTERFACE_0X0404_PROXY_H
#define HOZON_HMI_HMIAPADASDATASERVICEINTERFACE_0X0404_PROXY_H

#include "ara/com/internal/proxy/proxy_adapter.h"
#include "ara/com/internal/proxy/event_adapter.h"
#include "ara/com/internal/proxy/field_adapter.h"
#include "ara/com/internal/proxy/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/hmi/hmiapadasdataserviceinterface_0x0404_common.h"
#include <string>

namespace hozon {
namespace hmi {
namespace proxy {
namespace events {
}

namespace fields {
    using HMIAPA = ara::com::internal::proxy::field::FieldAdapter<::hozon::hmi::APA_Dataproperties_Struct>;
    using HMIHPP = ara::com::internal::proxy::field::FieldAdapter<::hozon::hmi::HPP_Path_Struct>;
    using HMINNS = ara::com::internal::proxy::field::FieldAdapter<::hozon::hmi::NNS_Info_Struct>;
    using HMIINS = ara::com::internal::proxy::field::FieldAdapter<::hozon::hmi::Ins_Info_Struct>;
    static constexpr ara::com::internal::EntityId HmiAPADASdataServiceInterface_0x0404HMIAPAId = 4027U; //HMIAPA_field_hash
    static constexpr ara::com::internal::EntityId HmiAPADASdataServiceInterface_0x0404HMIAPAGetterId = 23624U; //HMIAPA_getter_hash
    static constexpr ara::com::internal::EntityId HmiAPADASdataServiceInterface_0x0404HMIHPPId = 64490U; //HMIHPP_field_hash
    static constexpr ara::com::internal::EntityId HmiAPADASdataServiceInterface_0x0404HMIHPPGetterId = 3607U; //HMIHPP_getter_hash
    static constexpr ara::com::internal::EntityId HmiAPADASdataServiceInterface_0x0404HMINNSId = 2757U; //HMINNS_field_hash
    static constexpr ara::com::internal::EntityId HmiAPADASdataServiceInterface_0x0404HMINNSSetterId = 63474U; //HMINNS_setter_hash
    static constexpr ara::com::internal::EntityId HmiAPADASdataServiceInterface_0x0404HMIINSId = 38289U; //HMIINS_field_hash
    static constexpr ara::com::internal::EntityId HmiAPADASdataServiceInterface_0x0404HMIINSSetterId = 57256U; //HMIINS_setter_hash
}

namespace methods {

} // namespace methods

class HmiAPADASdataServiceInterface_0x0404Proxy {
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

    virtual ~HmiAPADASdataServiceInterface_0x0404Proxy()
    {
    }

    explicit HmiAPADASdataServiceInterface_0x0404Proxy(const vrtf::vcc::api::types::HandleType &handle)
        : proxyAdapter(std::make_unique<ara::com::internal::proxy::ProxyAdapter>(::hozon::hmi::HmiAPADASdataServiceInterface_0x0404::ServiceIdentifier, handle)),
          HMIAPA(proxyAdapter->GetProxy(), fields::HmiAPADASdataServiceInterface_0x0404HMIAPAId, proxyAdapter->GetHandle(), ::hozon::hmi::HmiAPADASdataServiceInterface_0x0404::ServiceIdentifier),
          HMIHPP(proxyAdapter->GetProxy(), fields::HmiAPADASdataServiceInterface_0x0404HMIHPPId, proxyAdapter->GetHandle(), ::hozon::hmi::HmiAPADASdataServiceInterface_0x0404::ServiceIdentifier),
          HMINNS(proxyAdapter->GetProxy(), fields::HmiAPADASdataServiceInterface_0x0404HMINNSId, proxyAdapter->GetHandle(), ::hozon::hmi::HmiAPADASdataServiceInterface_0x0404::ServiceIdentifier),
          HMIINS(proxyAdapter->GetProxy(), fields::HmiAPADASdataServiceInterface_0x0404HMIINSId, proxyAdapter->GetHandle(), ::hozon::hmi::HmiAPADASdataServiceInterface_0x0404::ServiceIdentifier) {
            HMIAPA.SetGetterEntityId(fields::HmiAPADASdataServiceInterface_0x0404HMIAPAGetterId);
            ara::core::Result<void> resultHMIAPA = proxyAdapter->InitializeField<::hozon::hmi::APA_Dataproperties_Struct>(HMIAPA);
            ThrowError(resultHMIAPA);
            HMIHPP.SetGetterEntityId(fields::HmiAPADASdataServiceInterface_0x0404HMIHPPGetterId);
            ara::core::Result<void> resultHMIHPP = proxyAdapter->InitializeField<::hozon::hmi::HPP_Path_Struct>(HMIHPP);
            ThrowError(resultHMIHPP);
            HMINNS.SetSetterEntityId(fields::HmiAPADASdataServiceInterface_0x0404HMINNSSetterId);
            ara::core::Result<void> resultHMINNS = proxyAdapter->InitializeField<::hozon::hmi::NNS_Info_Struct>(HMINNS);
            ThrowError(resultHMINNS);
            HMIINS.SetSetterEntityId(fields::HmiAPADASdataServiceInterface_0x0404HMIINSSetterId);
            ara::core::Result<void> resultHMIINS = proxyAdapter->InitializeField<::hozon::hmi::Ins_Info_Struct>(HMIINS);
            ThrowError(resultHMIINS);
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

    HmiAPADASdataServiceInterface_0x0404Proxy(const HmiAPADASdataServiceInterface_0x0404Proxy&) = delete;
    HmiAPADASdataServiceInterface_0x0404Proxy& operator=(const HmiAPADASdataServiceInterface_0x0404Proxy&) = delete;

    HmiAPADASdataServiceInterface_0x0404Proxy(HmiAPADASdataServiceInterface_0x0404Proxy&&) = default;
    HmiAPADASdataServiceInterface_0x0404Proxy& operator=(HmiAPADASdataServiceInterface_0x0404Proxy&&) = default;
    HmiAPADASdataServiceInterface_0x0404Proxy(ConstructionToken&& token) noexcept
        : proxyAdapter(token.GetProxyAdapter()),
          HMIAPA(proxyAdapter->GetProxy(), fields::HmiAPADASdataServiceInterface_0x0404HMIAPAId, proxyAdapter->GetHandle(), ::hozon::hmi::HmiAPADASdataServiceInterface_0x0404::ServiceIdentifier),
          HMIHPP(proxyAdapter->GetProxy(), fields::HmiAPADASdataServiceInterface_0x0404HMIHPPId, proxyAdapter->GetHandle(), ::hozon::hmi::HmiAPADASdataServiceInterface_0x0404::ServiceIdentifier),
          HMINNS(proxyAdapter->GetProxy(), fields::HmiAPADASdataServiceInterface_0x0404HMINNSId, proxyAdapter->GetHandle(), ::hozon::hmi::HmiAPADASdataServiceInterface_0x0404::ServiceIdentifier),
          HMIINS(proxyAdapter->GetProxy(), fields::HmiAPADASdataServiceInterface_0x0404HMIINSId, proxyAdapter->GetHandle(), ::hozon::hmi::HmiAPADASdataServiceInterface_0x0404::ServiceIdentifier) {
        HMIAPA.SetGetterEntityId(fields::HmiAPADASdataServiceInterface_0x0404HMIAPAGetterId);
        HMIHPP.SetGetterEntityId(fields::HmiAPADASdataServiceInterface_0x0404HMIHPPGetterId);
        HMINNS.SetSetterEntityId(fields::HmiAPADASdataServiceInterface_0x0404HMINNSSetterId);
        HMIINS.SetSetterEntityId(fields::HmiAPADASdataServiceInterface_0x0404HMIINSSetterId);
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        const vrtf::vcc::api::types::HandleType &handle)
    {
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> preProxyAdapter =
            std::make_unique<ara::com::internal::proxy::ProxyAdapter>(
               ::hozon::hmi::HmiAPADASdataServiceInterface_0x0404::ServiceIdentifier, handle);
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            fields::HMIAPA HMIAPA(preProxyAdapter->GetProxy(), fields::HmiAPADASdataServiceInterface_0x0404HMIAPAId, handle, ::hozon::hmi::HmiAPADASdataServiceInterface_0x0404::ServiceIdentifier);
            HMIAPA.SetGetterEntityId(fields::HmiAPADASdataServiceInterface_0x0404HMIAPAGetterId);
            initResult = preProxyAdapter->InitializeField<::hozon::hmi::APA_Dataproperties_Struct>(HMIAPA);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            fields::HMIHPP HMIHPP(preProxyAdapter->GetProxy(), fields::HmiAPADASdataServiceInterface_0x0404HMIHPPId, handle, ::hozon::hmi::HmiAPADASdataServiceInterface_0x0404::ServiceIdentifier);
            HMIHPP.SetGetterEntityId(fields::HmiAPADASdataServiceInterface_0x0404HMIHPPGetterId);
            initResult = preProxyAdapter->InitializeField<::hozon::hmi::HPP_Path_Struct>(HMIHPP);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            fields::HMINNS HMINNS(preProxyAdapter->GetProxy(), fields::HmiAPADASdataServiceInterface_0x0404HMINNSId, handle, ::hozon::hmi::HmiAPADASdataServiceInterface_0x0404::ServiceIdentifier);
            HMINNS.SetSetterEntityId(fields::HmiAPADASdataServiceInterface_0x0404HMINNSSetterId);
            initResult = preProxyAdapter->InitializeField<::hozon::hmi::NNS_Info_Struct>(HMINNS);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            fields::HMIINS HMIINS(preProxyAdapter->GetProxy(), fields::HmiAPADASdataServiceInterface_0x0404HMIINSId, handle, ::hozon::hmi::HmiAPADASdataServiceInterface_0x0404::ServiceIdentifier);
            HMIINS.SetSetterEntityId(fields::HmiAPADASdataServiceInterface_0x0404HMIINSSetterId);
            initResult = preProxyAdapter->InitializeField<::hozon::hmi::Ins_Info_Struct>(HMIINS);
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
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::hozon::hmi::HmiAPADASdataServiceInterface_0x0404::ServiceIdentifier, instance);
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType> handler,
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::hozon::hmi::HmiAPADASdataServiceInterface_0x0404::ServiceIdentifier, specifier);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::hozon::hmi::HmiAPADASdataServiceInterface_0x0404::ServiceIdentifier, instance);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::hozon::hmi::HmiAPADASdataServiceInterface_0x0404::ServiceIdentifier, specifier);
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
    fields::HMIAPA HMIAPA;
    fields::HMIHPP HMIHPP;
    fields::HMINNS HMINNS;
    fields::HMIINS HMIINS;
};
} // namespace proxy
} // namespace hmi
} // namespace hozon

#endif // HOZON_HMI_HMIAPADASDATASERVICEINTERFACE_0X0404_PROXY_H
