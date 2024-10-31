/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_SWM_UPDATEMANAGERSERVICEINTERFACE_PROXY_H
#define MDC_SWM_UPDATEMANAGERSERVICEINTERFACE_PROXY_H

#include "ara/com/internal/proxy/proxy_adapter.h"
#include "ara/com/internal/proxy/event_adapter.h"
#include "ara/com/internal/proxy/field_adapter.h"
#include "ara/com/internal/proxy/method_adapter.h"
#include "ara/com/crc_verification.h"
#include "mdc/swm/updatemanagerserviceinterface_common.h"
#include <string>

namespace mdc {
namespace swm {
namespace proxy {
namespace events {
}

namespace fields {
}

namespace methods {
static constexpr ara::com::internal::EntityId UpdateManagerServiceInterfaceActivateId = 4102U; //Activate_method_hash
static constexpr ara::com::internal::EntityId UpdateManagerServiceInterfaceFinishId = 43324U; //Finish_method_hash
static constexpr ara::com::internal::EntityId UpdateManagerServiceInterfaceGetActivationProgressId = 29393U; //GetActivationProgress_method_hash
static constexpr ara::com::internal::EntityId UpdateManagerServiceInterfaceGetSwProcessProgressId = 36314U; //GetSwProcessProgress_method_hash
static constexpr ara::com::internal::EntityId UpdateManagerServiceInterfaceGetUpdatePreCheckProgressId = 43431U; //GetUpdatePreCheckProgress_method_hash
static constexpr ara::com::internal::EntityId UpdateManagerServiceInterfaceGetUpdatePreCheckResultId = 45144U; //GetUpdatePreCheckResult_method_hash
static constexpr ara::com::internal::EntityId UpdateManagerServiceInterfaceGetUpdateProgressId = 34795U; //GetUpdateProgress_method_hash
static constexpr ara::com::internal::EntityId UpdateManagerServiceInterfaceGetUpdateStatusId = 61203U; //GetUpdateStatus_method_hash
static constexpr ara::com::internal::EntityId UpdateManagerServiceInterfaceProcessSwPackageId = 16077U; //ProcessSwPackage_method_hash
static constexpr ara::com::internal::EntityId UpdateManagerServiceInterfaceUpdateId = 64985U; //Update_method_hash
static constexpr ara::com::internal::EntityId UpdateManagerServiceInterfaceUpdatePreCheckId = 2020U; //UpdatePreCheck_method_hash
static constexpr ara::com::internal::EntityId UpdateManagerServiceInterfaceGetVerifyListId = 17584U; //GetVerifyList_method_hash
static constexpr ara::com::internal::EntityId UpdateManagerServiceInterfaceRollbackId = 28344U; //Rollback_method_hash
static constexpr ara::com::internal::EntityId UpdateManagerServiceInterfaceActivateByModeId = 2439U; //ActivateByMode_method_hash


class Activate {
public:
    using Output = mdc::swm::methods::Activate::Output;

    Activate(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::mdc::swm::SwNameVectorType& preActivate, const ::mdc::swm::SwNameVectorType& verify)
    {
        return method_(preActivate, verify);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::mdc::swm::SwNameVectorType, ::mdc::swm::SwNameVectorType> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::mdc::swm::SwNameVectorType, ::mdc::swm::SwNameVectorType> method_;
};

class Finish {
public:
    using Output = mdc::swm::methods::Finish::Output;

    Finish(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
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

class GetActivationProgress {
public:
    using Output = mdc::swm::methods::GetActivationProgress::Output;

    GetActivationProgress(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
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

class GetSwProcessProgress {
public:
    using Output = mdc::swm::methods::GetSwProcessProgress::Output;

    GetSwProcessProgress(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::mdc::swm::TransferIdType& id)
    {
        return method_(id);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::mdc::swm::TransferIdType> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::mdc::swm::TransferIdType> method_;
};

class GetUpdatePreCheckProgress {
public:
    using Output = mdc::swm::methods::GetUpdatePreCheckProgress::Output;

    GetUpdatePreCheckProgress(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
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

class GetUpdatePreCheckResult {
public:
    using Output = mdc::swm::methods::GetUpdatePreCheckResult::Output;

    GetUpdatePreCheckResult(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
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

class GetUpdateProgress {
public:
    using Output = mdc::swm::methods::GetUpdateProgress::Output;

    GetUpdateProgress(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
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

class GetUpdateStatus {
public:
    using Output = mdc::swm::methods::GetUpdateStatus::Output;

    GetUpdateStatus(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
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

class ProcessSwPackage {
public:
    using Output = mdc::swm::methods::ProcessSwPackage::Output;

    ProcessSwPackage(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::mdc::swm::TransferIdType& id)
    {
        return method_(id);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::mdc::swm::TransferIdType> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::mdc::swm::TransferIdType> method_;
};

class Update {
public:
    using Output = mdc::swm::methods::Update::Output;

    Update(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::String& packagePath, const ::Int8& mode)
    {
        return method_(packagePath, mode);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::Int8> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::String, ::Int8> method_;
};

class UpdatePreCheck {
public:
    using Output = mdc::swm::methods::UpdatePreCheck::Output;

    UpdatePreCheck(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::UInt8& mode, const ::String& packagePath)
    {
        return method_(mode, packagePath);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::UInt8, ::String> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::UInt8, ::String> method_;
};

class GetVerifyList {
public:
    using Output = mdc::swm::methods::GetVerifyList::Output;

    GetVerifyList(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
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

class Rollback {
public:
    using Output = mdc::swm::methods::Rollback::Output;

    Rollback(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::String& packagePath)
    {
        return method_(packagePath);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::String> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::String> method_;
};

class ActivateByMode {
public:
    using Output = mdc::swm::methods::ActivateByMode::Output;

    ActivateByMode(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId): method_(proxy, entityId){}
    void Initialize(const std::shared_ptr<vrtf::vcc::Proxy>& proxy, const ara::com::internal::EntityId entityId)
    {
        method_.Initialize(proxy, entityId);
    }
    ara::core::Future<Output> operator()(const ::mdc::swm::SwNameVectorType& preActivate, const ::mdc::swm::SwNameVectorType& verify, const ::Int8& mode)
    {
        return method_(preActivate, verify, mode);
    }

    ara::com::internal::proxy::method::MethodAdapter<Output, ::mdc::swm::SwNameVectorType, ::mdc::swm::SwNameVectorType, ::Int8> GetMethod() const
    {
        return method_;
    }

    ara::com::e2e::SMState GetSMState() const noexcept
    {
        return method_.GetSMState();
    }

private:
    ara::com::internal::proxy::method::MethodAdapter<Output, ::mdc::swm::SwNameVectorType, ::mdc::swm::SwNameVectorType, ::Int8> method_;
};
} // namespace methods

class UpdateManagerServiceInterfaceProxy {
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

    virtual ~UpdateManagerServiceInterfaceProxy()
    {
    }

    explicit UpdateManagerServiceInterfaceProxy(const vrtf::vcc::api::types::HandleType &handle)
        : proxyAdapter(std::make_unique<ara::com::internal::proxy::ProxyAdapter>(::mdc::swm::UpdateManagerServiceInterface::ServiceIdentifier, handle)),
          Activate(proxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceActivateId),
          Finish(proxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceFinishId),
          GetActivationProgress(proxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceGetActivationProgressId),
          GetSwProcessProgress(proxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceGetSwProcessProgressId),
          GetUpdatePreCheckProgress(proxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceGetUpdatePreCheckProgressId),
          GetUpdatePreCheckResult(proxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceGetUpdatePreCheckResultId),
          GetUpdateProgress(proxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceGetUpdateProgressId),
          GetUpdateStatus(proxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceGetUpdateStatusId),
          ProcessSwPackage(proxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceProcessSwPackageId),
          Update(proxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceUpdateId),
          UpdatePreCheck(proxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceUpdatePreCheckId),
          GetVerifyList(proxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceGetVerifyListId),
          Rollback(proxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceRollbackId),
          ActivateByMode(proxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceActivateByModeId){
            ara::core::Result<void> resultActivate = proxyAdapter->InitializeMethod<methods::Activate::Output>(methods::UpdateManagerServiceInterfaceActivateId);
            ThrowError(resultActivate);
            ara::core::Result<void> resultFinish = proxyAdapter->InitializeMethod<methods::Finish::Output>(methods::UpdateManagerServiceInterfaceFinishId);
            ThrowError(resultFinish);
            ara::core::Result<void> resultGetActivationProgress = proxyAdapter->InitializeMethod<methods::GetActivationProgress::Output>(methods::UpdateManagerServiceInterfaceGetActivationProgressId);
            ThrowError(resultGetActivationProgress);
            ara::core::Result<void> resultGetSwProcessProgress = proxyAdapter->InitializeMethod<methods::GetSwProcessProgress::Output>(methods::UpdateManagerServiceInterfaceGetSwProcessProgressId);
            ThrowError(resultGetSwProcessProgress);
            ara::core::Result<void> resultGetUpdatePreCheckProgress = proxyAdapter->InitializeMethod<methods::GetUpdatePreCheckProgress::Output>(methods::UpdateManagerServiceInterfaceGetUpdatePreCheckProgressId);
            ThrowError(resultGetUpdatePreCheckProgress);
            ara::core::Result<void> resultGetUpdatePreCheckResult = proxyAdapter->InitializeMethod<methods::GetUpdatePreCheckResult::Output>(methods::UpdateManagerServiceInterfaceGetUpdatePreCheckResultId);
            ThrowError(resultGetUpdatePreCheckResult);
            ara::core::Result<void> resultGetUpdateProgress = proxyAdapter->InitializeMethod<methods::GetUpdateProgress::Output>(methods::UpdateManagerServiceInterfaceGetUpdateProgressId);
            ThrowError(resultGetUpdateProgress);
            ara::core::Result<void> resultGetUpdateStatus = proxyAdapter->InitializeMethod<methods::GetUpdateStatus::Output>(methods::UpdateManagerServiceInterfaceGetUpdateStatusId);
            ThrowError(resultGetUpdateStatus);
            ara::core::Result<void> resultProcessSwPackage = proxyAdapter->InitializeMethod<methods::ProcessSwPackage::Output>(methods::UpdateManagerServiceInterfaceProcessSwPackageId);
            ThrowError(resultProcessSwPackage);
            ara::core::Result<void> resultUpdate = proxyAdapter->InitializeMethod<methods::Update::Output>(methods::UpdateManagerServiceInterfaceUpdateId);
            ThrowError(resultUpdate);
            ara::core::Result<void> resultUpdatePreCheck = proxyAdapter->InitializeMethod<methods::UpdatePreCheck::Output>(methods::UpdateManagerServiceInterfaceUpdatePreCheckId);
            ThrowError(resultUpdatePreCheck);
            ara::core::Result<void> resultGetVerifyList = proxyAdapter->InitializeMethod<methods::GetVerifyList::Output>(methods::UpdateManagerServiceInterfaceGetVerifyListId);
            ThrowError(resultGetVerifyList);
            ara::core::Result<void> resultRollback = proxyAdapter->InitializeMethod<methods::Rollback::Output>(methods::UpdateManagerServiceInterfaceRollbackId);
            ThrowError(resultRollback);
            ara::core::Result<void> resultActivateByMode = proxyAdapter->InitializeMethod<methods::ActivateByMode::Output>(methods::UpdateManagerServiceInterfaceActivateByModeId);
            ThrowError(resultActivateByMode);
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

    UpdateManagerServiceInterfaceProxy(const UpdateManagerServiceInterfaceProxy&) = delete;
    UpdateManagerServiceInterfaceProxy& operator=(const UpdateManagerServiceInterfaceProxy&) = delete;

    UpdateManagerServiceInterfaceProxy(UpdateManagerServiceInterfaceProxy&&) = default;
    UpdateManagerServiceInterfaceProxy& operator=(UpdateManagerServiceInterfaceProxy&&) = default;
    UpdateManagerServiceInterfaceProxy(ConstructionToken&& token) noexcept
        : proxyAdapter(token.GetProxyAdapter()),
          Activate(proxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceActivateId),
          Finish(proxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceFinishId),
          GetActivationProgress(proxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceGetActivationProgressId),
          GetSwProcessProgress(proxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceGetSwProcessProgressId),
          GetUpdatePreCheckProgress(proxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceGetUpdatePreCheckProgressId),
          GetUpdatePreCheckResult(proxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceGetUpdatePreCheckResultId),
          GetUpdateProgress(proxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceGetUpdateProgressId),
          GetUpdateStatus(proxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceGetUpdateStatusId),
          ProcessSwPackage(proxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceProcessSwPackageId),
          Update(proxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceUpdateId),
          UpdatePreCheck(proxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceUpdatePreCheckId),
          GetVerifyList(proxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceGetVerifyListId),
          Rollback(proxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceRollbackId),
          ActivateByMode(proxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceActivateByModeId){
    }

    static ara::core::Result<ConstructionToken> Preconstruct(
        const vrtf::vcc::api::types::HandleType &handle)
    {
        std::unique_ptr<ara::com::internal::proxy::ProxyAdapter> preProxyAdapter =
            std::make_unique<ara::com::internal::proxy::ProxyAdapter>(
               ::mdc::swm::UpdateManagerServiceInterface::ServiceIdentifier, handle);
        bool result = true;
        ara::core::Result<void> initResult;
        do {
            const methods::Activate Activate(preProxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceActivateId);
            initResult = preProxyAdapter->InitializeMethod<methods::Activate::Output>(methods::UpdateManagerServiceInterfaceActivateId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::Finish Finish(preProxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceFinishId);
            initResult = preProxyAdapter->InitializeMethod<methods::Finish::Output>(methods::UpdateManagerServiceInterfaceFinishId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::GetActivationProgress GetActivationProgress(preProxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceGetActivationProgressId);
            initResult = preProxyAdapter->InitializeMethod<methods::GetActivationProgress::Output>(methods::UpdateManagerServiceInterfaceGetActivationProgressId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::GetSwProcessProgress GetSwProcessProgress(preProxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceGetSwProcessProgressId);
            initResult = preProxyAdapter->InitializeMethod<methods::GetSwProcessProgress::Output>(methods::UpdateManagerServiceInterfaceGetSwProcessProgressId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::GetUpdatePreCheckProgress GetUpdatePreCheckProgress(preProxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceGetUpdatePreCheckProgressId);
            initResult = preProxyAdapter->InitializeMethod<methods::GetUpdatePreCheckProgress::Output>(methods::UpdateManagerServiceInterfaceGetUpdatePreCheckProgressId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::GetUpdatePreCheckResult GetUpdatePreCheckResult(preProxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceGetUpdatePreCheckResultId);
            initResult = preProxyAdapter->InitializeMethod<methods::GetUpdatePreCheckResult::Output>(methods::UpdateManagerServiceInterfaceGetUpdatePreCheckResultId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::GetUpdateProgress GetUpdateProgress(preProxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceGetUpdateProgressId);
            initResult = preProxyAdapter->InitializeMethod<methods::GetUpdateProgress::Output>(methods::UpdateManagerServiceInterfaceGetUpdateProgressId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::GetUpdateStatus GetUpdateStatus(preProxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceGetUpdateStatusId);
            initResult = preProxyAdapter->InitializeMethod<methods::GetUpdateStatus::Output>(methods::UpdateManagerServiceInterfaceGetUpdateStatusId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::ProcessSwPackage ProcessSwPackage(preProxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceProcessSwPackageId);
            initResult = preProxyAdapter->InitializeMethod<methods::ProcessSwPackage::Output>(methods::UpdateManagerServiceInterfaceProcessSwPackageId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::Update Update(preProxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceUpdateId);
            initResult = preProxyAdapter->InitializeMethod<methods::Update::Output>(methods::UpdateManagerServiceInterfaceUpdateId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::UpdatePreCheck UpdatePreCheck(preProxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceUpdatePreCheckId);
            initResult = preProxyAdapter->InitializeMethod<methods::UpdatePreCheck::Output>(methods::UpdateManagerServiceInterfaceUpdatePreCheckId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::GetVerifyList GetVerifyList(preProxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceGetVerifyListId);
            initResult = preProxyAdapter->InitializeMethod<methods::GetVerifyList::Output>(methods::UpdateManagerServiceInterfaceGetVerifyListId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::Rollback Rollback(preProxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceRollbackId);
            initResult = preProxyAdapter->InitializeMethod<methods::Rollback::Output>(methods::UpdateManagerServiceInterfaceRollbackId);
            if (!initResult.HasValue()) {
                result = false;
                break;
            }
            const methods::ActivateByMode ActivateByMode(preProxyAdapter->GetProxy(), methods::UpdateManagerServiceInterfaceActivateByModeId);
            initResult = preProxyAdapter->InitializeMethod<methods::ActivateByMode::Output>(methods::UpdateManagerServiceInterfaceActivateByModeId);
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
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::mdc::swm::UpdateManagerServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::FindServiceHandle StartFindService(
        const ara::com::FindServiceHandler<ara::com::internal::proxy::ProxyAdapter::HandleType> handler,
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::StartFindService(handler, ::mdc::swm::UpdateManagerServiceInterface::ServiceIdentifier, specifier);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::com::InstanceIdentifier instance)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::mdc::swm::UpdateManagerServiceInterface::ServiceIdentifier, instance);
    }

    static ara::com::ServiceHandleContainer<ara::com::internal::proxy::ProxyAdapter::HandleType> FindService(
        const ara::core::InstanceSpecifier specifier)
    {
        return ara::com::internal::proxy::ProxyAdapter::FindService(::mdc::swm::UpdateManagerServiceInterface::ServiceIdentifier, specifier);
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
    methods::Activate Activate;
    methods::Finish Finish;
    methods::GetActivationProgress GetActivationProgress;
    methods::GetSwProcessProgress GetSwProcessProgress;
    methods::GetUpdatePreCheckProgress GetUpdatePreCheckProgress;
    methods::GetUpdatePreCheckResult GetUpdatePreCheckResult;
    methods::GetUpdateProgress GetUpdateProgress;
    methods::GetUpdateStatus GetUpdateStatus;
    methods::ProcessSwPackage ProcessSwPackage;
    methods::Update Update;
    methods::UpdatePreCheck UpdatePreCheck;
    methods::GetVerifyList GetVerifyList;
    methods::Rollback Rollback;
    methods::ActivateByMode ActivateByMode;
};
} // namespace proxy
} // namespace swm
} // namespace mdc

#endif // MDC_SWM_UPDATEMANAGERSERVICEINTERFACE_PROXY_H
