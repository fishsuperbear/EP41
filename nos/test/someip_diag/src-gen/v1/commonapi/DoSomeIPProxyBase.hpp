/*
* This file was generated by the CommonAPI Generators.
* Used org.genivi.commonapi.core 3.2.0.v202012010850.
* Used org.franca.core 0.13.1.201807231814.
*
* This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
* If a copy of the MPL was not distributed with this file, You can obtain one at
* http://mozilla.org/MPL/2.0/.
*/
#ifndef V1_COMMONAPI_Do_Some_IP_PROXY_BASE_HPP_
#define V1_COMMONAPI_Do_Some_IP_PROXY_BASE_HPP_

#include <v1/commonapi/DoSomeIP.hpp>



#if !defined (COMMONAPI_INTERNAL_COMPILATION)
#define COMMONAPI_INTERNAL_COMPILATION
#define HAS_DEFINED_COMMONAPI_INTERNAL_COMPILATION_HERE
#endif

#include <CommonAPI/Deployment.hpp>
#include <CommonAPI/InputStream.hpp>
#include <CommonAPI/OutputStream.hpp>
#include <CommonAPI/Struct.hpp>
#include <cstdint>
#include <vector>

#include <CommonAPI/Proxy.hpp>
#include <functional>
#include <future>

#if defined (HAS_DEFINED_COMMONAPI_INTERNAL_COMPILATION_HERE)
#undef COMMONAPI_INTERNAL_COMPILATION
#undef HAS_DEFINED_COMMONAPI_INTERNAL_COMPILATION_HERE
#endif

namespace v1 {
namespace commonapi {

class DoSomeIPProxyBase
    : virtual public CommonAPI::Proxy {
public:

    typedef std::function<void(const CommonAPI::CallStatus&, const DoSomeIP::stdErrorTypeEnum&, const DoSomeIP::DoSomeIPRespUdsMessage&)> UdsMessageRequestAsyncCallback;

    virtual void udsMessageRequest(DoSomeIP::DoSomeIPReqUdsMessage _req, CommonAPI::CallStatus &_internalCallStatus, DoSomeIP::stdErrorTypeEnum &_error, DoSomeIP::DoSomeIPRespUdsMessage &_resp, const CommonAPI::CallInfo *_info = nullptr) = 0;
    virtual std::future<CommonAPI::CallStatus> udsMessageRequestAsync(const DoSomeIP::DoSomeIPReqUdsMessage &_req, UdsMessageRequestAsyncCallback _callback = nullptr, const CommonAPI::CallInfo *_info = nullptr) = 0;

    virtual std::future<void> getCompletionFuture() = 0;
};

} // namespace commonapi
} // namespace v1


// Compatibility
namespace v1_0 = v1;

#endif // V1_COMMONAPI_Do_Some_IP_PROXY_BASE_HPP_