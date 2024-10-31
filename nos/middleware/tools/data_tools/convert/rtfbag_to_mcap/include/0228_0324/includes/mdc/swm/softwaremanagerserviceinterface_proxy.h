/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_SWM_SOFTWAREMANAGERSERVICEINTERFACE_PROXY_H
#define MDC_SWM_SOFTWAREMANAGERSERVICEINTERFACE_PROXY_H

#include "ara/com/internal/proxy/proxy_adapter.h"
#include "ara/com/internal/proxy/event_adapter.h"
#include "ara/com/internal/proxy/field_adapter.h"
#include "ara/com/internal/proxy/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "mdc/swm/softwaremanagerserviceinterface_common.h"
#include <string>

namespace mdc {
namespace swm {
namespace proxy {
namespace events {
}

namespace fields {
}

namespace methods {
static constexpr ara::com::internal::EntityId SoftwareManagerServiceInterfaceGetFinggerPrintId = 10835U; //GetFinggerPrint_method_hash
static constexpr ara::com::internal::EntityId SoftwareManagerServiceInterfaceGetHistoryId = 23435U; //GetHistory_method_hash
static constexpr ara::com::internal::EntityId SoftwareManagerServiceInterfaceGetHistoryInfoId = 4639U; //GetHistoryInfo_method_hash
static constexpr ara::com::internal::EntityId SoftwareManagerServiceInterfaceGetSwClusterInfoId = 25213U; //GetSwClusterInfo_method_hash
static constexpr ara::com::internal::EntityId SoftwareManagerServiceInterfaceGetVersionInfoId = 26771U; //GetVersionInfo_method_hash
static constexpr ara::com::internal::EntityId SoftwareManagerServiceInterfaceSetFinggerPrintId = 22609U; //SetFinggerPrint_method_hash
static constexpr ara::com::internal::EntityId SoftwareManagerServiceInterfaceGetSpecificVersionInfoId = 32010U; //GetSpecificVersionInfo_method_hash
static constexpr ara::com::internal::EntityId SoftwareManagerServiceInterfaceRefreshVersionId = 11944U; //RefreshVersion_method_hash
static constexpr ara::com::internal::EntityId SoftwareManagerServiceInterfaceGetUpdateLogListId = 38286U; //GetUpdateLogList_method_hash


class GetFinggerPrint {
public:
    using Output = mdc::swm::methods::GetFinggerPrint::Output;

    GetFinggerPrint(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
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

class GetHistory {
public:
    using Output = mdc::swm::methods::GetHistory::Output;

    GetHistory(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::UInt64& timestampGE, const ::UInt64& timestampLT)
    {
        return method_(timestampGE, timestampLT);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::UInt64, ::UInt64> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::UInt64, ::UInt64> method_;
};

class GetHistoryInfo {
public:
    using Output = mdc::swm::methods::GetHistoryInfo::Output;

    GetHistoryInfo(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
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

class GetSwClusterInfo {
public:
    using Output = mdc::swm::methods::GetSwClusterInfo::Output;

    GetSwClusterInfo(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
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

class GetVersionInfo {
public:
    using Output = mdc::swm::methods::GetVersionInfo::Output;

    GetVersionInfo(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
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

class SetFinggerPrint {
public:
    using Output = mdc::swm::methods::SetFinggerPrint::Output;

    SetFinggerPrint(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::mdc::swm::FinggerPrintType& fingerPrint)
    {
        return method_(fingerPrint);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::mdc::swm::FinggerPrintType> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::mdc::swm::FinggerPrintType> method_;
};

class GetSpecificVersionInfo {
public:
    using Output = mdc::swm::methods::GetSpecificVersionInfo::Output;

    GetSpecificVersionInfo(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::String& location, const ::String& name)
    {
        return method_(location, name);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::String> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::String> method_;
};

class RefreshVersion {
public:
    using Output = mdc::swm::methods::RefreshVersion::Output;

    RefreshVersion(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::Boolean& isSync)
    {
        return method_(isSync);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::Boolean> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::Boolean> method_;
};

class GetUpdateLogList {
public:
    using Output = mdc::swm::methods::GetUpdateLogList::Output;

    GetUpdateLogList(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
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
} // namespace methods

class SoftwareManagerServiceInterfaceProxy {
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

    virtual ~SoftwareManagerServiceInterfaceProxy()
    {
    }

    explicit SoftwareManagerServiceInterfaceProxy(const vrtf::vcc::api::types::HandleType &handle)
        : proxyAdapter(std::make_unique<ara::com::internal::proxy::ProxyAdapter>(::mdc::swm::SoftwareManagerServiceInterface::ServiceIdentifier, handle)),
          GetFinggerPrint(proxyAdapter->GetProxy(), methods::SoftwareManagerServiceInterfaceGetFinggerPrintId),
          GetHistory(proxyAdapter->GetProxy(), methods::SoftwareManagerServiceInterfaceGetHistoryId),
          GetHistoryInfo(proxyAdapter->GetProxy(), methods::SoftwareManagerServiceInterfaceGetHistoryInfoId),
          GetSwClusterInfo(proxyAdapter->GetProxy(), methods::SoftwareManagerServiceInterfaceGetSwClusterInfoId),
          GetVersionInfo(proxyAdapter->GetProxy(), methods::SoftwareManagerServiceInterfaceGetVersionInfoId),
          SetFinggerPrint(proxyAdapter->GetProxy(), methods::SoftwareManagerServiceInterfaceSetFinggerPrintId),
          GetSpecificVersionInfo(proxyAdapter->GetProxy(), methods::SoftwareManagerServiceInterfaceGetSpecificVersionInfoId),
          RefreshVersion(proxyAdapter->GetProxy(), methods::SoftwareManagerServiceInterfaceRefreshVersionId),
          GetUpdateLogList(proxyAdapter->GetProxy(), methods::SoftwareManagerServiceInterfaceGetUpdateLogListId){
            ara::core::Result<void> resultGetFinggerPrint = proxyAdapter->InitializeMethod<methods::GetFinggerPrint::Output>(methods::SoftwareManagerServiceInterfaceGetFinggerPrintId);
            ThrowError(resultGetFinggerPrint);
            ara::core::Result<void> resultGetHistory = proxyAdapter->InitializeMethod<methods::GetHistory::Output>(methods::SoftwareManagerServiceInterfaceGetHistoryId);
            ThrowError(resultGetHistory);
            ara::core::Result<void> resultGetHistoryInfo = proxyAdapter->InitializeMethod<methods::GetHistoryInfo::Output>(methods::SoftwareManagerServiceInterfaceGetHistoryInfoId);
            ThrowError(resultGetHistoryInfo);
            ara::core::Result<void> resultGetSwClusterInfo = proxyAdapter->InitializeMethod<methods::GetSwClusterInfo::Output>(methods::SoftwareManagerServiceInterfaceGetSwClusterInfoId);
            ThrowError(resultGetSwClusterInfo);
            ara::core::Result<void> resultGetVersionInfo = proxyAdapter->InitializeMethod<methods::GetVersionInfo::Output>(methods::SoftwareManagerServiceInterfaceGetVersionInfoId);
            ThrowError(resultGetVersionInfo);
            ara::core::Result<void> resultSetFinggerPrint = proxyAdapter->InitializeMethod<methods::SetFinggerPrint::Output>(methods::SoftwareManagerServiceInterfaceSetFinggerPrintId);
            ThrowError(resultSetFinggerPrint);
            ara::core::Result<void> resultGetSpecificVersionInfo = proxyAdapter->InitializeMethod<methods::GetSpecificVersionInfo::Output>(methods::SoftwareManagerServiceInterfaceGetSpecificVersionInfoId);
            ThrowError(resultGetSpecificVersionInfo);
            ara::core::Result<void> resultRefreshVersion = proxyAdapter->InitializeMethod<methods::RefreshVersion::Output>(methods::SoftwareManagerServiceInterfaceRefreshVersionId);
            ThrowError(resultRefreshVersion);
            ara::core::Result<void> resultGetUpdateLogList = proxyAdapter->InitializeMethod<methods::GetUpdateLogList::Output>(methods::SoftwareManagerServiceInterfaceGetUpdateLogListId);
            ThrowError(resultGetUpdateLogList);
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

    SoftwareManagerServiceInterfaceProxy(const SoftwareManagerServiceInterfaceProxy&) = delete;
    SoftwareManagerServiceInterfaceProxy& operator=(const SoftwareManagerServiceInterfaceProxy&) = delete;

    SoftwareManagerServiceInterfaceProxy(SoftwareManagerServiceInterfaceProxy&&) = default;
    SoftwareManagerServiceInterfaceProxy& operator=(SoftwareManagerServiceInterfaceProxy&&) = default;
    SoftwareManagerServiceInterfaceProxy(ConstructionToken&& token) noexcept
        : proxyAdapter(token.GetProxyAdapter()),
          GetFinggerPrint(proxyAdapter->GetProxy(), methods::SoftwareManagerServiceInterfaceGetFinggerPrintId),
          GetHistory(proxyAdapter->GetProxy(), methods::SoftwareManagerServiceInterfaceGetHistoryId),
          GetHistoryInfo(proxyAdapter->GetProxy(), methods::SoftwareManagerServiceInterfaceGetHistoryInfoId),
          GetSwClusterInfo(proxyAdapter->GetProxy(), methods::SoftwareManagerServiceInterfaceGetSwClusterInfoId),
          GetVersionInfo(proxyAdapter->GetProxy(), methods::SoftwareManagerServiceInterfaceGetVersionInfoId),
          SetFinggerPrint(proxyAdapter->GetProxy(), methods::SoftwareManagerServiceInterfaceSetFinggerPrintId),
          GetSpecificVersionInfo(proxyAdapter->GetProxy(), methods::SoftwareManagerServiceInterfaceGetSpecificVersionInfoId),
          RefreshVersion(proxyAdapter->GetProxy(), methods::SoftwareManagerServiceInterfaceRefreshVersionId),
          GetUpdateLogList(proxyAdapter->GetProxy(), methods::SoftwareManagerServiceInterfaceGetUpdateLogListId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        const vrtf::vcc::api::types::HandleType &handle)
    {
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> preProxyAdapter =
            std::make_unique<ara::com::internal::proxy::ProxyAdapter>(
               ::mdc::swm::SoftwareManagerServiceInterface::ServiceIdentifier, handle);
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            const methods::GetFinggerPrint GetFinggerPrint(preProxyAdapter->GetProxy(), methods::SoftwareManagerServiceInterfaceGetFinggerPrintId);
            initResult = preProxyAdapter->InitializeMethod<methods::GetFinggerPrint::Output>(methods::SoftwareManagerServiceInterfaceGetFinggerPrintId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::GetHistory GetHistory(preProxyAdapter->GetProxy(), methods::SoftwareManagerServiceInterfaceGetHistoryId);
            initResult = preProxyAdapter->InitializeMethod<methods::GetHistory::Output>(methods::SoftwareManagerServiceInterfaceGetHistoryId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::GetHistoryInfo GetHistoryInfo(preProxyAdapter->GetProxy(), methods::SoftwareManagerServiceInterfaceGetHistoryInfoId);
            initResult = preProxyAdapter->InitializeMethod<methods::GetHistoryInfo::Output>(methods::SoftwareManagerServiceInterfaceGetHistoryInfoId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::GetSwClusterInfo GetSwClusterInfo(preProxyAdapter->GetProxy(), methods::SoftwareManagerServiceInterfaceGetSwClusterInfoId);
            initResult = preProxyAdapter->InitializeMethod<methods::GetSwClusterInfo::Output>(methods::SoftwareManagerServiceInterfaceGetSwClusterInfoId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::GetVersionInfo GetVersionInfo(preProxyAdapter->GetProxy(), methods::SoftwareManagerServiceInterfaceGetVersionInfoId);
            initResult = preProxyAdapter->InitializeMethod<methods::GetVersionInfo::Output>(methods::SoftwareManagerServiceInterfaceGetVersionInfoId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::SetFinggerPrint SetFinggerPrint(preProxyAdapter->GetProxy(), methods::SoftwareManagerServiceInterfaceSetFinggerPrintId);
            initResult = preProxyAdapter->InitializeMethod<methods::SetFinggerPrint::Output>(methods::SoftwareManagerServiceInterfaceSetFinggerPrintId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::GetSpecificVersionInfo GetSpecificVersionInfo(preProxyAdapter->GetProxy(), methods::SoftwareManagerServiceInterfaceGetSpecificVersionInfoId);
            initResult = preProxyAdapter->InitializeMethod<methods::GetSpecificVersionInfo::Output>(methods::SoftwareManagerServiceInterfaceGetSpecificVersionInfoId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::RefreshVersion RefreshVersion(preProxyAdapter->GetProxy(), methods::SoftwareManagerServiceInterfaceRefreshVersionId);
            initResult = preProxyAdapter->InitializeMethod<methods::RefreshVersion::Output>(methods::SoftwareManagerServiceInterfaceRefreshVersionId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::GetUpdateLogList GetUpdateLogList(preProxyAdapter->GetProxy(), methods::SoftwareManagerServiceInterfaceGetUpdateLogListId);
            initResult = preProxyAdapter->InitializeMethod<methods::GetUpdateLogList::Output>(methods::SoftwareManagerServiceInterfaceGetUpdateLogListId);
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
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::mdc::swm::SoftwareManagerServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType> handler,
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::mdc::swm::SoftwareManagerServiceInterface::ServiceIdentifier, specifier);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::mdc::swm::SoftwareManagerServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::mdc::swm::SoftwareManagerServiceInterface::ServiceIdentifier, specifier);
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
    methods::GetFinggerPrint GetFinggerPrint;
    methods::GetHistory GetHistory;
    methods::GetHistoryInfo GetHistoryInfo;
    methods::GetSwClusterInfo GetSwClusterInfo;
    methods::GetVersionInfo GetVersionInfo;
    methods::SetFinggerPrint SetFinggerPrint;
    methods::GetSpecificVersionInfo GetSpecificVersionInfo;
    methods::RefreshVersion RefreshVersion;
    methods::GetUpdateLogList GetUpdateLogList;
};
} // namespace proxy
} // namespace swm
} // namespace mdc

#endif // MDC_SWM_SOFTWAREMANAGERSERVICEINTERFACE_PROXY_H
