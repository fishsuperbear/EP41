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
#ifndef EXTEND_E2E_E2EPROTECT_NE_SOMEIP_E2E_PROTECT_H_
#define EXTEND_E2E_E2EPROTECT_NE_SOMEIP_E2E_PROTECT_H_

#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <vector>
#include "extend/e2exf/e2e_handler.h"
#include "extend/e2exf/transformer.h"

    /**
    * @brief This class contains the API that can be used to protect data with SOME/IP Event and Method in a polling mode.
    *
    * This class contains the API that can be used to protect data with SOME/IP Event and Method in a polling mode.
    */
    class NESomeIPE2EProtect {
    public:
        NESomeIPE2EProtect();
        virtual ~NESomeIPE2EProtect();

        static NESomeIPE2EProtect& instance();

        bool configure(const char* bindingConfigurationPath, uint16_t bindingConfigLen,
                       const char* e2exfConfigurationPath, uint16_t e2exfConfigLen);
        bool configure(const std::string& bindingConfigurationPath, e2exf::ConfigurationFormat bindingConfigurationFormat,
                       const std::string& e2exfConfigurationPath, e2exf::ConfigurationFormat e2exfConfigurationFormat);

        bool isProtected(const std::uint16_t serviceId, const std::uint16_t instanceId, const std::uint16_t eventOrMethodId);

        e2e::Result handleCheckStatus(std::uint16_t serviceId, std::uint16_t instanceId, std::uint16_t eventId,
                                      E2E_state_machine::E2ECheckStatus checkStatus);

        bool protectEvent(const std::uint16_t serviceId, const std::uint16_t instanceId, const std::uint16_t eventId,
                          const crc::Buffer& inputBuffer, crc::Buffer& outputBuffer);

        e2e::Result checkEvent(const std::uint16_t serviceId, const std::uint16_t instanceId, const std::uint16_t eventId,
                               const crc::Buffer& inputBuffer, crc::Buffer& outputBuffer,
                               std::uint32_t& dataId, std::uint32_t& counter);

        bool protectMethod(const std::uint16_t serviceId, const std::uint16_t instanceId, const std::uint16_t methodId,
                           const crc::Buffer& inputBuffer, crc::Buffer& outputBuffer);

        e2e::Result checkMethod(const std::uint16_t serviceId, const std::uint16_t instanceId, const std::uint16_t methodId,
                                const crc::Buffer& inputBuffer, crc::Buffer& outputBuffer,
                                std::uint32_t& dataId, std::uint32_t& counter);

        bool someipE2EProtect(std::uint16_t serviceId, std::uint16_t instanceId, std::uint16_t eventOrMethodId,
                              std::string* someipData, bool isEvent);

        bool someipE2ECheck(std::uint16_t serviceId, std::uint16_t instanceId, std::uint16_t eventOrMethodId,
                            std::string* someipData, bool isEvent, std::shared_ptr<e2e::Result>& retCheck,
                            std::uint32_t& dataId, std::uint32_t& counter);

    };

#endif  // EXTEND_E2E_E2EPROTECT_NE_SOMEIP_E2E_PROTECT_H_
/* EOF */
