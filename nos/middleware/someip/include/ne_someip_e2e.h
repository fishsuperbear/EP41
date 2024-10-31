/**

* Copyright @ 2020 - 2027 iAuto Software(Shanghai) Co., Ltd.

* All Rights Reserved.

*

* Redistribution and use in source and binary forms, with or without

* modification, are NOT permitted except as agreed by

* iAuto Software(Shanghai) Co., Ltd.

*

* Unless required by applicable law or agreed to in writing, software

* distributed under the License is distributed on an "AS IS" BASIS,

* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

*/
#ifndef INCLUDE_NE_SOMEIP_E2E_H_
#define INCLUDE_NE_SOMEIP_E2E_H_

#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <memory>
#include "ne_someip_define.h"
#include "ne_someip_e2e_result.h"
#include "ne_someip_e2e_state_machine.h"

    /**
     * @brief This class contains the public API of the SOME/IP e2e implementation.
     *
     * This class provides interface to e2e protect for SOME/IP message, the functions include are configure, protect and check.
     */
    class NESomeIPE2E {
    public:
        NESomeIPE2E();
        ~NESomeIPE2E();

        /**
         * @brief load e2e config if support E2E Protect.
         *
         * @param [in] bindingConfigurationPath : The E2E configure of E2E abd dataId Mapping.
         * @param [in] e2exfConfigurationPath : The E2E configure of E2E state machines.
         *
         * @return NESomeIPErrorCode_Ok indicates success, other value indicates failure.
         *
         * @attention Synchronous I/F.
         */
        ne_someip_error_code_t loadE2EConfig(const std::string& bindingConfigurationPath, const std::string& e2exfConfigurationPath);

        /**
         * @brief get whether E2E protect is configured of event or method.
         *
         * @param [in] serviceId : The identifier of the service.
         * @param [in] instanceId : The identifier of the service instance.
         * @param [in] eventId : The identifier of the service event or method.
         * @param [out] isProtected : the return value of e2e protect.
         *
         * @return NESomeIPErrorCode_Ok indicates success, other value indicates failure.
         *
         * @attention Synchronous I/F.
         */
        ne_someip_error_code_t isProtected(const ne_someip_service_id_t serviceId,
            const ne_someip_instance_id_t instanceId,
            const ne_someip_method_id_t eventOrMethodId,
            bool& isProtected);

        /**
         * @brief E2E prottect for someip message.
         *
         * @param [in] header : The SOME/IP message header.
         * @param [in] payload : The SOME/IP message payload.
         *
         * @return NESomeIPErrorCode_Ok indicates success, other value indicates failure.
         *
         * @attention Synchronous I/F.
         */
        ne_someip_error_code_t e2eProtect(ne_someip_header_t& header, ne_someip_instance_id_t instanceId, ne_someip_payload_t& payload);

        /**
         * @brief E2E Check for someip message.
         *
         * @param [in] header : The SOME/IP message header.
         * @param [in] payload : The SOME/IP message payload.
         * @param [out] retCheck : The e2e check result.
         * @param [out] dataId : The e2e dataId of the message.
         * @param [out] counter : The e2e counter of the message.
         *
         * @return NESomeIPErrorCode_Ok indicates success, other value indicates failure.
         *
         * @attention Synchronous I/F.
         */
        ne_someip_error_code_t e2eCheck(ne_someip_header_t& header, ne_someip_instance_id_t instanceId, ne_someip_payload_t& payload,
            std::shared_ptr<e2e::Result>& retCheck, std::uint32_t& dataId, std::uint32_t& counter);

        /**
         * @brief handle Check Status.
         *
         * @param [in] serviceId : The identifier of the service.
         * @param [in] instanceId : The identifier of the service instance.
         * @param [in] eventId : The identifier of the service event/field.
         * @param [in] checkStatus : The e2e check status.
         * @param [out] retCheck : The e2e check result.
         *
         * @return NESomeIPErrorCode_Ok indicates success, other value indicates failure.
         *
         * @attention Synchronous I/F.
         */
        ne_someip_error_code_t handleCheckStatus(ne_someip_service_id_t serviceId,
            ne_someip_instance_id_t instanceId,
            ne_someip_method_id_t eventId,
            E2E_state_machine::E2ECheckStatus checkStatus,
            std::shared_ptr<e2e::Result>& retCheck);

    };
#endif  // INCLUDE_NE_SOMEIP_E2E_H_
/* EOF */
