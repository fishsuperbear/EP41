/*
 * This file was generated by the CommonAPI Generators.
 * Used org.genivi.commonapi.someip 3.2.0.v202012010944.
 * Used org.franca.core 0.13.1.201807231814.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
 * If a copy of the MPL was not distributed with this file, You can obtain one at
 * http://mozilla.org/MPL/2.0/.
 */
#include <v1/commonapi/DoSomeIPSomeIPProxy.hpp>

#if !defined (COMMONAPI_INTERNAL_COMPILATION)
#define COMMONAPI_INTERNAL_COMPILATION
#define HAS_DEFINED_COMMONAPI_INTERNAL_COMPILATION_HERE
#endif

#include <CommonAPI/SomeIP/AddressTranslator.hpp>

#if defined (HAS_DEFINED_COMMONAPI_INTERNAL_COMPILATION_HERE)
#undef COMMONAPI_INTERNAL_COMPILATION
#undef HAS_DEFINED_COMMONAPI_INTERNAL_COMPILATION_HERE
#endif

namespace v1 {
namespace commonapi {

std::shared_ptr<CommonAPI::SomeIP::Proxy> createDoSomeIPSomeIPProxy(
    const CommonAPI::SomeIP::Address &_address,
    const std::shared_ptr<CommonAPI::SomeIP::ProxyConnection> &_connection) {
    return std::make_shared< DoSomeIPSomeIPProxy>(_address, _connection);
}

void initializeDoSomeIPSomeIPProxy() {
    CommonAPI::SomeIP::AddressTranslator::get()->insert(
        "local:commonapi.DoSomeIP:v1_0:commonapi.dosomeip",
        0x1234, 0x5678, 1, 0);
    CommonAPI::SomeIP::Factory::get()->registerProxyCreateMethod(
        "commonapi.DoSomeIP:v1_0",
        &createDoSomeIPSomeIPProxy);
}

INITIALIZER(registerDoSomeIPSomeIPProxy) {
    CommonAPI::SomeIP::Factory::get()->registerInterface(initializeDoSomeIPSomeIPProxy);
}

DoSomeIPSomeIPProxy::DoSomeIPSomeIPProxy(
    const CommonAPI::SomeIP::Address &_address,
    const std::shared_ptr<CommonAPI::SomeIP::ProxyConnection> &_connection)
        : CommonAPI::SomeIP::Proxy(_address, _connection)
{
}

DoSomeIPSomeIPProxy::~DoSomeIPSomeIPProxy() {
    completed_.set_value();
}



void DoSomeIPSomeIPProxy::udsMessageRequest(DoSomeIP::DoSomeIPReqUdsMessage _req, CommonAPI::CallStatus &_internalCallStatus, DoSomeIP::stdErrorTypeEnum &_error, DoSomeIP::DoSomeIPRespUdsMessage &_resp, const CommonAPI::CallInfo *_info) {
    CommonAPI::Deployable< DoSomeIP::stdErrorTypeEnum, ::v1::commonapi::DoSomeIP_::stdErrorTypeEnumDeployment_t> deploy_error(&::v1::commonapi::DoSomeIP_::stdErrorTypeEnumDeployment);
    CommonAPI::Deployable< DoSomeIP::DoSomeIPReqUdsMessage, ::v1::commonapi::DoSomeIP_::DoSomeIPReqUdsMessageDeployment_t> deploy_req(_req, &::v1::commonapi::DoSomeIP_::DoSomeIPReqUdsMessageDeployment);
    CommonAPI::Deployable< DoSomeIP::DoSomeIPRespUdsMessage, ::v1::commonapi::DoSomeIP_::DoSomeIPRespUdsMessageDeployment_t> deploy_resp(&::v1::commonapi::DoSomeIP_::DoSomeIPRespUdsMessageDeployment);
    CommonAPI::SomeIP::ProxyHelper<
        CommonAPI::SomeIP::SerializableArguments<
            CommonAPI::Deployable<
                DoSomeIP::DoSomeIPReqUdsMessage,
                ::v1::commonapi::DoSomeIP_::DoSomeIPReqUdsMessageDeployment_t
            >
        >,
        CommonAPI::SomeIP::SerializableArguments<
            CommonAPI::Deployable<
                DoSomeIP::stdErrorTypeEnum,
                ::v1::commonapi::DoSomeIP_::stdErrorTypeEnumDeployment_t
            >,
            CommonAPI::Deployable<
                DoSomeIP::DoSomeIPRespUdsMessage,
                ::v1::commonapi::DoSomeIP_::DoSomeIPRespUdsMessageDeployment_t
            >
        >
    >::callMethodWithReply(
        *this,
        CommonAPI::SomeIP::method_id_t(0x7530),
        false,
        false,
        (_info ? _info : &CommonAPI::SomeIP::defaultCallInfo),
        deploy_req,
        _internalCallStatus,
        deploy_error,
        deploy_resp);
    _error = deploy_error.getValue();
    _resp = deploy_resp.getValue();
}

std::future<CommonAPI::CallStatus> DoSomeIPSomeIPProxy::udsMessageRequestAsync(const DoSomeIP::DoSomeIPReqUdsMessage &_req, UdsMessageRequestAsyncCallback _callback, const CommonAPI::CallInfo *_info) {
    CommonAPI::Deployable< DoSomeIP::stdErrorTypeEnum, ::v1::commonapi::DoSomeIP_::stdErrorTypeEnumDeployment_t> deploy_error(&::v1::commonapi::DoSomeIP_::stdErrorTypeEnumDeployment);
    CommonAPI::Deployable< DoSomeIP::DoSomeIPReqUdsMessage, ::v1::commonapi::DoSomeIP_::DoSomeIPReqUdsMessageDeployment_t> deploy_req(_req, &::v1::commonapi::DoSomeIP_::DoSomeIPReqUdsMessageDeployment);
    CommonAPI::Deployable< DoSomeIP::DoSomeIPRespUdsMessage, ::v1::commonapi::DoSomeIP_::DoSomeIPRespUdsMessageDeployment_t> deploy_resp(&::v1::commonapi::DoSomeIP_::DoSomeIPRespUdsMessageDeployment);
    return CommonAPI::SomeIP::ProxyHelper<
        CommonAPI::SomeIP::SerializableArguments<
            CommonAPI::Deployable<
                DoSomeIP::DoSomeIPReqUdsMessage,
                ::v1::commonapi::DoSomeIP_::DoSomeIPReqUdsMessageDeployment_t
            >
        >,
        CommonAPI::SomeIP::SerializableArguments<
            CommonAPI::Deployable<
                DoSomeIP::stdErrorTypeEnum,
                ::v1::commonapi::DoSomeIP_::stdErrorTypeEnumDeployment_t
            >,
            CommonAPI::Deployable<
                DoSomeIP::DoSomeIPRespUdsMessage,
                ::v1::commonapi::DoSomeIP_::DoSomeIPRespUdsMessageDeployment_t
            >
        >
    >::callMethodAsync(
        *this,
        CommonAPI::SomeIP::method_id_t(0x7530),
        false,
        false,
        (_info ? _info : &CommonAPI::SomeIP::defaultCallInfo),
        deploy_req,
        [_callback] (CommonAPI::CallStatus _internalCallStatus, CommonAPI::Deployable< DoSomeIP::stdErrorTypeEnum, ::v1::commonapi::DoSomeIP_::stdErrorTypeEnumDeployment_t > _deploy_error, CommonAPI::Deployable< DoSomeIP::DoSomeIPRespUdsMessage, ::v1::commonapi::DoSomeIP_::DoSomeIPRespUdsMessageDeployment_t > _resp) {
            if (_callback)
                _callback(_internalCallStatus, _deploy_error.getValue(), _resp.getValue());
        },
        std::make_tuple(deploy_error, deploy_resp));
}

void DoSomeIPSomeIPProxy::getOwnVersion(uint16_t& ownVersionMajor, uint16_t& ownVersionMinor) const {
    ownVersionMajor = 1;
    ownVersionMinor = 0;
}

std::future<void> DoSomeIPSomeIPProxy::getCompletionFuture() {
    return completed_.get_future();
}

} // namespace commonapi
} // namespace v1
