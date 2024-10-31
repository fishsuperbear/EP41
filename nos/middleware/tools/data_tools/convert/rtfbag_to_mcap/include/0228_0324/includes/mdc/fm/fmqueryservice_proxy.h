/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_FM_FMQUERYSERVICE_PROXY_H
#define MDC_FM_FMQUERYSERVICE_PROXY_H

#include "ara/com/internal/proxy/proxy_adapter.h"
#include "ara/com/internal/proxy/event_adapter.h"
#include "ara/com/internal/proxy/field_adapter.h"
#include "ara/com/internal/proxy/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "mdc/fm/fmqueryservice_common.h"
#include <string>

namespace mdc {
namespace fm {
namespace proxy {
namespace events {
}

namespace fields {
}

namespace methods {
static constexpr ara::com::internal::EntityId FmQueryServiceQueryFaultDetailId = 36459U; //QueryFaultDetail_method_hash
static constexpr ara::com::internal::EntityId FmQueryServiceQueryFaultOnFlagId = 43267U; //QueryFaultOnFlag_method_hash
static constexpr ara::com::internal::EntityId FmQueryServiceQueryFaultStatisticId = 26435U; //QueryFaultStatistic_method_hash


class QueryFaultDetail {
public:
    using Output = mdc::fm::methods::QueryFaultDetail::Output;

    QueryFaultDetail(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
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

class QueryFaultOnFlag {
public:
    using Output = mdc::fm::methods::QueryFaultOnFlag::Output;

    QueryFaultOnFlag(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::UInt32& flag)
    {
        return method_(flag);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::UInt32> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::UInt32> method_;
};

class QueryFaultStatistic {
public:
    using Output = mdc::fm::methods::QueryFaultStatistic::Output;

    QueryFaultStatistic(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::UInt32& flag)
    {
        return method_(flag);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::UInt32> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::UInt32> method_;
};
} // namespace methods

class FmQueryServiceProxy {
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

    virtual ~FmQueryServiceProxy()
    {
    }

    explicit FmQueryServiceProxy(const vrtf::vcc::api::types::HandleType &handle)
        : proxyAdapter(std::make_unique<ara::com::internal::proxy::ProxyAdapter>(::mdc::fm::FmQueryService::ServiceIdentifier, handle)),
          QueryFaultDetail(proxyAdapter->GetProxy(), methods::FmQueryServiceQueryFaultDetailId),
          QueryFaultOnFlag(proxyAdapter->GetProxy(), methods::FmQueryServiceQueryFaultOnFlagId),
          QueryFaultStatistic(proxyAdapter->GetProxy(), methods::FmQueryServiceQueryFaultStatisticId){
            ara::core::Result<void> resultQueryFaultDetail = proxyAdapter->InitializeMethod<methods::QueryFaultDetail::Output>(methods::FmQueryServiceQueryFaultDetailId);
            ThrowError(resultQueryFaultDetail);
            ara::core::Result<void> resultQueryFaultOnFlag = proxyAdapter->InitializeMethod<methods::QueryFaultOnFlag::Output>(methods::FmQueryServiceQueryFaultOnFlagId);
            ThrowError(resultQueryFaultOnFlag);
            ara::core::Result<void> resultQueryFaultStatistic = proxyAdapter->InitializeMethod<methods::QueryFaultStatistic::Output>(methods::FmQueryServiceQueryFaultStatisticId);
            ThrowError(resultQueryFaultStatistic);
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

    FmQueryServiceProxy(const FmQueryServiceProxy&) = delete;
    FmQueryServiceProxy& operator=(const FmQueryServiceProxy&) = delete;

    FmQueryServiceProxy(FmQueryServiceProxy&&) = default;
    FmQueryServiceProxy& operator=(FmQueryServiceProxy&&) = default;
    FmQueryServiceProxy(ConstructionToken&& token) noexcept
        : proxyAdapter(token.GetProxyAdapter()),
          QueryFaultDetail(proxyAdapter->GetProxy(), methods::FmQueryServiceQueryFaultDetailId),
          QueryFaultOnFlag(proxyAdapter->GetProxy(), methods::FmQueryServiceQueryFaultOnFlagId),
          QueryFaultStatistic(proxyAdapter->GetProxy(), methods::FmQueryServiceQueryFaultStatisticId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        const vrtf::vcc::api::types::HandleType &handle)
    {
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> preProxyAdapter =
            std::make_unique<ara::com::internal::proxy::ProxyAdapter>(
               ::mdc::fm::FmQueryService::ServiceIdentifier, handle);
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            const methods::QueryFaultDetail QueryFaultDetail(preProxyAdapter->GetProxy(), methods::FmQueryServiceQueryFaultDetailId);
            initResult = preProxyAdapter->InitializeMethod<methods::QueryFaultDetail::Output>(methods::FmQueryServiceQueryFaultDetailId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::QueryFaultOnFlag QueryFaultOnFlag(preProxyAdapter->GetProxy(), methods::FmQueryServiceQueryFaultOnFlagId);
            initResult = preProxyAdapter->InitializeMethod<methods::QueryFaultOnFlag::Output>(methods::FmQueryServiceQueryFaultOnFlagId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::QueryFaultStatistic QueryFaultStatistic(preProxyAdapter->GetProxy(), methods::FmQueryServiceQueryFaultStatisticId);
            initResult = preProxyAdapter->InitializeMethod<methods::QueryFaultStatistic::Output>(methods::FmQueryServiceQueryFaultStatisticId);
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
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::mdc::fm::FmQueryService::ServiceIdentifier, instance);
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType> handler,
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::mdc::fm::FmQueryService::ServiceIdentifier, specifier);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::mdc::fm::FmQueryService::ServiceIdentifier, instance);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::mdc::fm::FmQueryService::ServiceIdentifier, specifier);
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
    methods::QueryFaultDetail QueryFaultDetail;
    methods::QueryFaultOnFlag QueryFaultOnFlag;
    methods::QueryFaultStatistic QueryFaultStatistic;
};
} // namespace proxy
} // namespace fm
} // namespace mdc

#endif // MDC_FM_FMQUERYSERVICE_PROXY_H
