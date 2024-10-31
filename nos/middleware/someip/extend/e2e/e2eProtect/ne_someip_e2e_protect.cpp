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
#include <cstddef>
#include <memory>
#include "ne_someip_log.h"
#include "ne_someip_e2e_protect.h"

    NESomeIPE2EProtect::NESomeIPE2EProtect()
    {
    }

    NESomeIPE2EProtect::~NESomeIPE2EProtect()
    {
    }

    NESomeIPE2EProtect& NESomeIPE2EProtect::instance() {
        static NESomeIPE2EProtect theInstance;
        return theInstance;
    }

    bool
    NESomeIPE2EProtect::configure(const char* bindingConfigurationPath, uint16_t bindingConfigLen,
                                  const char* e2exfConfigurationPath, uint16_t e2exfConfigLen)
    {
        const std::string bindingConfig(bindingConfigurationPath, bindingConfigLen);
        const std::string e2exfConfig(e2exfConfigurationPath, e2exfConfigLen);
        return e2exf::E2EHandler::Configure(bindingConfig, e2exf::ConfigurationFormat::JSON,
                                            e2exfConfig, e2exf::ConfigurationFormat::JSON);
    }

    bool
    NESomeIPE2EProtect::configure(const std::string& bindingConfigurationPath, e2exf::ConfigurationFormat bindingConfigurationFormat,
                                  const std::string& e2exfConfigurationPath, e2exf::ConfigurationFormat e2exfConfigurationFormat)
    {
        return e2exf::E2EHandler::Configure(bindingConfigurationPath, bindingConfigurationFormat,
                                            e2exfConfigurationPath, e2exfConfigurationFormat);
    }

    bool
    NESomeIPE2EProtect::isProtected(const std::uint16_t serviceId, const std::uint16_t instanceId, const std::uint16_t eventOrMethodId) {
        return e2exf::E2EHandler::IsProtected(serviceId, instanceId, eventOrMethodId);
    }

    e2e::Result
    NESomeIPE2EProtect::handleCheckStatus(std::uint16_t serviceId, std::uint16_t instanceId, std::uint16_t eventId,
                                          E2E_state_machine::E2ECheckStatus checkStatus)
    {
        return e2exf::E2EHandler::HandleCheckStatus(serviceId, instanceId, eventId, checkStatus);
    }

    bool
    NESomeIPE2EProtect::protectEvent(const std::uint16_t serviceId, const std::uint16_t instanceId, const std::uint16_t eventId,
                                     const crc::Buffer& inputBuffer, crc::Buffer& outputBuffer)
    {
        return e2exf::E2EHandler::ProtectEvent(serviceId, instanceId, eventId, inputBuffer, outputBuffer);
    }

    e2e::Result
    NESomeIPE2EProtect::checkEvent(const std::uint16_t serviceId, const std::uint16_t instanceId, const std::uint16_t eventId,
                                   const crc::Buffer& inputBuffer, crc::Buffer& outputBuffer,
                                   std::uint32_t& dataId, std::uint32_t& counter)
    {
        e2exf::E2EHandler::GetDataIdAndCounter(serviceId, instanceId, eventId, inputBuffer, dataId, counter);
        return e2exf::E2EHandler::CheckEvent(serviceId, instanceId, eventId, inputBuffer, outputBuffer);
    }

    bool
    NESomeIPE2EProtect::protectMethod(const std::uint16_t serviceId, const std::uint16_t instanceId, const std::uint16_t methodId,
                                      const crc::Buffer& inputBuffer, crc::Buffer& outputBuffer)
    {
        return e2exf::E2EHandler::ProtectMethod(serviceId, instanceId, methodId, inputBuffer, outputBuffer);
    }

    e2e::Result
    NESomeIPE2EProtect::checkMethod(const std::uint16_t serviceId, const std::uint16_t instanceId, const std::uint16_t methodId,
                                    const crc::Buffer& inputBuffer, crc::Buffer& outputBuffer,
                                    std::uint32_t& dataId, std::uint32_t& counter)
    {
        e2exf::E2EHandler::GetDataIdAndCounter(serviceId, instanceId, methodId, inputBuffer, dataId, counter);
        return e2exf::E2EHandler::CheckMethod(serviceId, instanceId, methodId, inputBuffer, outputBuffer);
    }

    bool
    NESomeIPE2EProtect::someipE2EProtect(std::uint16_t serviceId, std::uint16_t instanceId, std::uint16_t eventOrMethodId,
                                         std::string* someipData, bool isEvent)
    {
        ne_someip_log_debug("start: serviceId:[0x%x] instanceId:[0x%x] eventOrMethodId:[0x%x] size:[%d]",
            serviceId, instanceId, eventOrMethodId, someipData->size());
        if (nullptr == someipData) {
            ne_someip_log_error("someipData null, error.");
            return false;
        }

        crc::Buffer inputBuffer(someipData->begin(), someipData->end());
        crc::Buffer outputBuffer;
        if (isEvent) {
            if(!protectEvent(serviceId, instanceId, eventOrMethodId, inputBuffer, outputBuffer)) {
                ne_someip_log_error("ProtectEvent error, serviceId:[0x%x] instanceId:[0x%x] eventId:[0x%x].", serviceId, instanceId, eventOrMethodId);
                return false;
            }
        }
        else {
            if(!protectMethod(serviceId, instanceId, eventOrMethodId, inputBuffer, outputBuffer)) {
                ne_someip_log_error("ProtectMethod error, serviceId:[0x%x] instanceId:[0x%x] MethodId:[0x%x].", serviceId, instanceId, eventOrMethodId);
                return false;
            }
        }

        std::string dataProtect(outputBuffer.begin(), outputBuffer.end());
        *someipData = dataProtect;
        ne_someip_log_debug("end: serviceId:[0x%x] instanceId:[0x%x] eventOrMethodId:[0x%x] size:[%d]",
            serviceId, instanceId, eventOrMethodId, someipData->size());

        return true;
    }

    bool
    NESomeIPE2EProtect::someipE2ECheck(std::uint16_t serviceId, std::uint16_t instanceId, std::uint16_t eventOrMethodId,
                                       std::string* someipData, bool isEvent, std::shared_ptr<e2e::Result>& retCheck,
                                       std::uint32_t& dataId, std::uint32_t& counter)
    {
        ne_someip_log_debug("start: serviceId:[0x%x] instanceId:[0x%x] eventOrMethodId:[0x%x] size:[%d]",
            serviceId, instanceId, eventOrMethodId, someipData->size());
        if (nullptr == someipData) {
            ne_someip_log_error("someipData null, error.");
            return false;
        }
        if (nullptr == retCheck) {
            ne_someip_log_error("retCheck null, error.");
            return false;
        }

        crc::Buffer inputBuffer(someipData->begin(), someipData->end());
        crc::Buffer outputBuffer;
        if (isEvent) {
            *retCheck = checkEvent(serviceId, instanceId, eventOrMethodId, inputBuffer, outputBuffer, dataId, counter);
        }
        else {
            *retCheck = checkMethod(serviceId, instanceId, eventOrMethodId, inputBuffer, outputBuffer, dataId, counter);
        }

        ne_someip_log_debug("someipE2ECheck IsOK:[%d], E2EState:[%d] ProfileCheckStatus:[%d]", retCheck->IsOK(),
            static_cast<uint32_t>(retCheck->GetSMState()), static_cast<uint32_t>(retCheck->GetProfileCheckStatus()));

        std::string dataProtect(outputBuffer.begin(), outputBuffer.end());
        *someipData = dataProtect;
        ne_someip_log_debug("end: serviceId:[0x%x] instanceId:[0x%x] eventOrMethodId:[0x%x] size:[%d]",
            serviceId, instanceId, eventOrMethodId, someipData->size());

        return true;
    }

/* EOF */
