/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_INTERFACE_DATACOLLECT_HOZONINTERFACE_DATACOLLECT_PROXY_H
#define HOZON_INTERFACE_DATACOLLECT_HOZONINTERFACE_DATACOLLECT_PROXY_H

#include "ara/com/internal/proxy/proxy_adapter.h"
#include "ara/com/internal/proxy/event_adapter.h"
#include "ara/com/internal/proxy/field_adapter.h"
#include "ara/com/internal/proxy/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "hozon/interface/datacollect/hozoninterface_datacollect_common.h"
#include <string>

namespace hozon {
namespace interface {
namespace datacollect {
namespace proxy {
namespace events {
}

namespace fields {
}

namespace methods {
static constexpr ara::com::internal::EntityId HozonInterface_DataCollectCollectTriggerReqId = 28344U; //CollectTriggerReq_method_hash
static constexpr ara::com::internal::EntityId HozonInterface_DataCollectCollectCustomDataReqId = 41558U; //CollectCustomDataReq_method_hash


class CollectTriggerReq {
public:
    using Output = void;

    CollectTriggerReq(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    void operator()(const ::hozon::datacollect::CollectTrigger& collectTrigger)
    {
        method_(collectTrigger);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::hozon::datacollect::CollectTrigger> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::hozon::datacollect::CollectTrigger> method_;
};

class CollectCustomDataReq {
public:
    using Output = void;

    CollectCustomDataReq(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    void operator()(const ::hozon::datacollect::CustomCollectData& customData)
    {
        method_(customData);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::hozon::datacollect::CustomCollectData> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::hozon::datacollect::CustomCollectData> method_;
};
} // namespace methods

class HozonInterface_DataCollectProxy {
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

    virtual ~HozonInterface_DataCollectProxy()
    {
    }

    explicit HozonInterface_DataCollectProxy(const vrtf::vcc::api::types::HandleType &handle)
        : proxyAdapter(std::make_unique<ara::com::internal::proxy::ProxyAdapter>(::hozon::interface::datacollect::HozonInterface_DataCollect::ServiceIdentifier, handle)),
          CollectTriggerReq(proxyAdapter->GetProxy(), methods::HozonInterface_DataCollectCollectTriggerReqId),
          CollectCustomDataReq(proxyAdapter->GetProxy(), methods::HozonInterface_DataCollectCollectCustomDataReqId){
            ara::core::Result<void> resultCollectTriggerReq = proxyAdapter->InitializeMethod<methods::CollectTriggerReq::Output>(methods::HozonInterface_DataCollectCollectTriggerReqId);
            ThrowError(resultCollectTriggerReq);
            ara::core::Result<void> resultCollectCustomDataReq = proxyAdapter->InitializeMethod<methods::CollectCustomDataReq::Output>(methods::HozonInterface_DataCollectCollectCustomDataReqId);
            ThrowError(resultCollectCustomDataReq);
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

    HozonInterface_DataCollectProxy(const HozonInterface_DataCollectProxy&) = delete;
    HozonInterface_DataCollectProxy& operator=(const HozonInterface_DataCollectProxy&) = delete;

    HozonInterface_DataCollectProxy(HozonInterface_DataCollectProxy&&) = default;
    HozonInterface_DataCollectProxy& operator=(HozonInterface_DataCollectProxy&&) = default;
    HozonInterface_DataCollectProxy(ConstructionToken&& token) noexcept
        : proxyAdapter(token.GetProxyAdapter()),
          CollectTriggerReq(proxyAdapter->GetProxy(), methods::HozonInterface_DataCollectCollectTriggerReqId),
          CollectCustomDataReq(proxyAdapter->GetProxy(), methods::HozonInterface_DataCollectCollectCustomDataReqId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        const vrtf::vcc::api::types::HandleType &handle)
    {
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> preProxyAdapter =
            std::make_unique<ara::com::internal::proxy::ProxyAdapter>(
               ::hozon::interface::datacollect::HozonInterface_DataCollect::ServiceIdentifier, handle);
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            const methods::CollectTriggerReq CollectTriggerReq(preProxyAdapter->GetProxy(), methods::HozonInterface_DataCollectCollectTriggerReqId);
            initResult = preProxyAdapter->InitializeMethod<methods::CollectTriggerReq::Output>(methods::HozonInterface_DataCollectCollectTriggerReqId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::CollectCustomDataReq CollectCustomDataReq(preProxyAdapter->GetProxy(), methods::HozonInterface_DataCollectCollectCustomDataReqId);
            initResult = preProxyAdapter->InitializeMethod<methods::CollectCustomDataReq::Output>(methods::HozonInterface_DataCollectCollectCustomDataReqId);
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
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::hozon::interface::datacollect::HozonInterface_DataCollect::ServiceIdentifier, instance);
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType> handler,
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::hozon::interface::datacollect::HozonInterface_DataCollect::ServiceIdentifier, specifier);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::hozon::interface::datacollect::HozonInterface_DataCollect::ServiceIdentifier, instance);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::hozon::interface::datacollect::HozonInterface_DataCollect::ServiceIdentifier, specifier);
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
    methods::CollectTriggerReq CollectTriggerReq;
    methods::CollectCustomDataReq CollectCustomDataReq;
};
} // namespace proxy
} // namespace datacollect
} // namespace interface
} // namespace hozon

#endif // HOZON_INTERFACE_DATACOLLECT_HOZONINTERFACE_DATACOLLECT_PROXY_H
