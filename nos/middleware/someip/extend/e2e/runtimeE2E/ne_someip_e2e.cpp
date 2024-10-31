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
#include <string.h>
#include <stdlib.h>
#include "ne_someip_log.h"
#include "ne_someip_e2e_protect.h"
#include "ne_someip_e2e.h"

    NESomeIPE2E::NESomeIPE2E() {
    }

    NESomeIPE2E::~NESomeIPE2E() {
    }

    ne_someip_error_code_t
    NESomeIPE2E::loadE2EConfig(const std::string& bindingConfigurationPath, const std::string& e2exfConfigurationPath)
    {
        ne_someip_log_debug("start, bindingConfigurationPath:[%s]  e2exfConfigurationPath:[%s]",
            bindingConfigurationPath.c_str(), e2exfConfigurationPath.c_str());

        bool ret = NESomeIPE2EProtect::instance().configure(
            bindingConfigurationPath.c_str(), bindingConfigurationPath.size(),
            e2exfConfigurationPath.c_str(), e2exfConfigurationPath.size());
        if (!ret) {
            ne_someip_log_debug("E2E configure error");
            return ne_someip_error_code_failed;
        }
        return ne_someip_error_code_ok;
    }

    ne_someip_error_code_t
    NESomeIPE2E::isProtected(const ne_someip_service_id_t serviceId,
            const ne_someip_instance_id_t instanceId,
            const ne_someip_method_id_t eventOrMethodId,
            bool& isProtected)
    {
        ne_someip_log_debug("start, serviceId:[%d] instanceId:[%d] eventOrMethodId:[%d]", serviceId, instanceId, eventOrMethodId);
        isProtected = NESomeIPE2EProtect::instance().isProtected(serviceId, instanceId, eventOrMethodId);
        ne_someip_log_debug("isProtectedd:[%d]", isProtected);
        return ne_someip_error_code_ok;
    }

    ne_someip_error_code_t
    NESomeIPE2E::e2eProtect(ne_someip_header_t& header, ne_someip_instance_id_t instanceId, ne_someip_payload_t& payload)
    {
        ne_someip_log_debug("start");
        ne_someip_method_id_t eventOrMethodId = header.method_id;
        if (eventOrMethodId & 0x8000) {
            ne_someip_log_error("event or method id [%d] wasn't in 0~32767.", eventOrMethodId);
            return ne_someip_error_code_failed;
        }
        ne_someip_service_id_t serviceId = header.service_id;
        ne_someip_log_debug("serviceId:[%d]  instanceId:[%d] eventOrMethodId:[%d]", serviceId, instanceId, eventOrMethodId);

        if (NESomeIPE2EProtect::instance().isProtected(serviceId, instanceId, eventOrMethodId)) {
            // get inputBuffer for E2EProtect.
            uint32_t totalLen = NESOMEIP_DEFAULT_MESSAGE_LENGTH;
            if (nullptr != payload.buffer_list && nullptr != *payload.buffer_list && 0 < payload.num) {
                for (uint32_t i = 0; i < payload.num; i++) {
                    totalLen += payload.buffer_list[i]->length;
                }
            }

            char* command = (char*)calloc(totalLen, sizeof(char));
            if (nullptr == command) {
                ne_someip_log_error("calloc error");
                return ne_someip_error_code_failed;
            }
            char* tempBuf = command;
            uint32_t length = 0;

            ne_someip_client_id_t clientId = header.client_id;
            memcpy(tempBuf, &clientId, sizeof(clientId));
            tempBuf = tempBuf + sizeof(clientId);
            length = length + sizeof(clientId);

            ne_someip_session_id_t sessionId = header.session_id;
            memcpy(tempBuf, &sessionId, sizeof(sessionId));
            tempBuf = tempBuf + sizeof(sessionId);
            length = length + sizeof(sessionId);

            ne_someip_protocol_version_t protocolVersion = header.protocol_version;
            memcpy(tempBuf, &protocolVersion, sizeof(protocolVersion));
            tempBuf = tempBuf + sizeof(protocolVersion);
            length = length + sizeof(protocolVersion);

            ne_someip_interface_version_t interfaceVersion = header.interface_version;
            memcpy(tempBuf, &interfaceVersion, sizeof(interfaceVersion));
            tempBuf = tempBuf + sizeof(interfaceVersion);
            length = length + sizeof(interfaceVersion);

            ne_someip_message_type_t messageType = header.message_type;
            memcpy(tempBuf, &messageType, sizeof(messageType));
            tempBuf = tempBuf + sizeof(messageType);
            length = length + sizeof(messageType);

            ne_someip_return_code_t returnCode = header.return_code;
            memcpy(tempBuf, &returnCode, sizeof(returnCode));
            tempBuf = tempBuf + sizeof(returnCode);
            length = length + sizeof(returnCode);

            ne_someip_log_debug("service id [0x%x], method id [0x%x], length [%d], clientId [0x%x], sessionId [%d]",
                serviceId, eventOrMethodId, header.message_length, clientId, sessionId);
            ne_someip_log_debug("protocol version [0x%x], interface version [0x%x], message type [0x%x], return code [0x%x]",
                protocolVersion, interfaceVersion, messageType, returnCode);

            if (nullptr != payload.buffer_list && nullptr != *payload.buffer_list && 0 < payload.num) {
                for (uint32_t i = 0; i < payload.num; i++) {
                    ne_someip_message_length_t payloadLen = payload.buffer_list[i]->length;
                    memcpy(tempBuf, payload.buffer_list[i]->data, payloadLen);
                    tempBuf = tempBuf + payloadLen;
                    length = length + payloadLen;
                }
            }

            std::string someipMessage(command, length);
            if (command) {
                free(command);
                command = nullptr;
            }
            ne_someip_log_debug("someipE2EProtect start: someipMessage.size:[%d]", someipMessage.size());
            bool isEvent = false;
            if (ne_someip_message_type_notification == messageType) {
                isEvent = true;
            }
            bool ret = NESomeIPE2EProtect::instance().someipE2EProtect(serviceId, instanceId, eventOrMethodId, &someipMessage, isEvent);
            if (!ret || someipMessage.size() < header.message_length) {
                ne_someip_log_error("someipE2EProtect error");
                return ne_someip_error_code_failed;
            }
            ne_someip_log_debug("someipE2EProtect end: someipMessage.size:[%d]", someipMessage.size());

            // check offset for someip
            ne_someip_client_id_t tempClientId;
            ne_someip_session_id_t tempSessionId;
            ne_someip_protocol_version_t tempProtocolVersion;
            ne_someip_interface_version_t tempInterfaceVersion;
            ne_someip_message_type_t tempMessageType;
            ne_someip_return_code_t tempReturnCode;
            uint32_t pos = 0;
            memcpy(&tempClientId, someipMessage.c_str() + pos, sizeof(tempClientId));
            pos += sizeof(tempClientId);
            memcpy(&tempSessionId, someipMessage.c_str() + pos, sizeof(tempSessionId));
            pos += sizeof(tempSessionId);
            memcpy(&tempProtocolVersion, someipMessage.c_str() + pos, sizeof(tempProtocolVersion));
            pos += sizeof(tempProtocolVersion);
            memcpy(&tempInterfaceVersion, someipMessage.c_str() + pos, sizeof(tempInterfaceVersion));
            pos += sizeof(tempInterfaceVersion);
            memcpy(&tempMessageType, someipMessage.c_str() + pos, sizeof(tempMessageType));
            pos += sizeof(tempMessageType);
            memcpy(&tempReturnCode, someipMessage.c_str() + pos, sizeof(tempReturnCode));
            pos += sizeof(tempReturnCode);
            ne_someip_log_debug("someipE2EProtect end: clientId:[%d], sessionId:[%d], protocolVersion:[%d], interfaceVersion:[%d], \
                messageType:[%d], returnCode:[%d], protectPayloadLength:[%d]",
                tempClientId, tempSessionId, tempProtocolVersion, tempInterfaceVersion, tempMessageType, tempReturnCode, (someipMessage.size() - pos));

            if (tempClientId != clientId ||
                tempSessionId != sessionId ||
                tempProtocolVersion != protocolVersion ||
                tempInterfaceVersion != interfaceVersion ||
                tempMessageType != messageType ||
                tempReturnCode != returnCode) {
                ne_someip_log_error("E2E profile offset error");
                return ne_someip_error_code_failed;
            }

            // TODO 这里默认payload.num为1
            if (payload.buffer_list[0]->data) {
                free(payload.buffer_list[0]->data);
            }
            uint32_t dataLengthProtected = someipMessage.size();
            payload.buffer_list[0]->length = dataLengthProtected - pos;
            header.message_length = dataLengthProtected;

            payload.buffer_list[0]->data = (uint8_t*)malloc(dataLengthProtected - pos);
            memcpy(payload.buffer_list[0]->data, someipMessage.c_str() + pos, dataLengthProtected - pos);
            return ne_someip_error_code_ok;
        } else {
            ne_someip_log_debug("serviceId[%d] instanceId[%d] eventOrMethodId[%d] did not E2E Protect in E2E configure",
                serviceId, instanceId, eventOrMethodId);
            return ne_someip_error_code_no_e2e_protect;
        }
    }

    ne_someip_error_code_t
    NESomeIPE2E::e2eCheck(ne_someip_header_t& header, ne_someip_instance_id_t instanceId, ne_someip_payload_t& payload,
            std::shared_ptr<e2e::Result>& retCheck, std::uint32_t& dataId, std::uint32_t& counter)
    {
        ne_someip_log_debug("start");
        if (nullptr == retCheck) {
            ne_someip_log_error("retCheck NULL");
            return ne_someip_error_code_failed;
        }

        ne_someip_method_id_t eventOrMethodId = header.method_id;
        if (eventOrMethodId & 0x8000) {
            ne_someip_log_error("event or method id [%d] wasn't in 0~32767.", eventOrMethodId);
            return ne_someip_error_code_failed;
        }

        ne_someip_service_id_t serviceId = header.service_id;
        ne_someip_log_debug("serviceId:[%d]  instanceId:[%d] eventOrMethodId:[%d]", serviceId, instanceId, eventOrMethodId);
        if (NESomeIPE2EProtect::instance().isProtected(serviceId, instanceId, eventOrMethodId)) {
            // get inputBuffer for E2ECheck.
            uint32_t totalLen = NESOMEIP_DEFAULT_MESSAGE_LENGTH;
            if (nullptr != payload.buffer_list && nullptr != *payload.buffer_list && 0 < payload.num) {
                for (uint32_t i = 0; i < payload.num; i++) {
                    totalLen += payload.buffer_list[i]->length;
                }
            }

            char* command = (char*)calloc(totalLen, sizeof(char));
            if (nullptr == command) {
                ne_someip_log_error("calloc error");
                return ne_someip_error_code_failed;
            }
            char* tempBuf = command;
            uint32_t length = 0;

            ne_someip_client_id_t clientId = header.client_id;
            memcpy(tempBuf, &clientId, sizeof(clientId));
            tempBuf = tempBuf + sizeof(clientId);
            length = length + sizeof(clientId);

            ne_someip_session_id_t sessionId = header.session_id;
            memcpy(tempBuf, &sessionId, sizeof(sessionId));
            tempBuf = tempBuf + sizeof(sessionId);
            length = length + sizeof(sessionId);

            ne_someip_protocol_version_t protocolVersion = header.protocol_version;
            memcpy(tempBuf, &protocolVersion, sizeof(protocolVersion));
            tempBuf = tempBuf + sizeof(protocolVersion);
            length = length + sizeof(protocolVersion);

            ne_someip_interface_version_t interfaceVersion = header.interface_version;
            memcpy(tempBuf, &interfaceVersion, sizeof(interfaceVersion));
            tempBuf = tempBuf + sizeof(interfaceVersion);
            length = length + sizeof(interfaceVersion);

            ne_someip_message_type_t messageType = header.message_type;
            memcpy(tempBuf, &messageType, sizeof(messageType));
            tempBuf = tempBuf + sizeof(messageType);
            length = length + sizeof(messageType);

            ne_someip_return_code_t returnCode = header.return_code;
            memcpy(tempBuf, &returnCode, sizeof(returnCode));
            tempBuf = tempBuf + sizeof(returnCode);
            length = length + sizeof(returnCode);

            ne_someip_log_debug("service id [0x%x], method id [0x%x], length [%d], clientId [0x%x], sessionId [%d]",
                serviceId, eventOrMethodId, header.message_length, clientId, sessionId);
            ne_someip_log_debug("protocol version [0x%x], interface version [0x%x], message type [0x%x], return code [0x%x]",
                protocolVersion, interfaceVersion, messageType, returnCode);

            if (nullptr != payload.buffer_list && nullptr != *payload.buffer_list && 0 < payload.num) {
                for (uint32_t i = 0; i < payload.num; i++) {
                    ne_someip_message_length_t payloadLen = payload.buffer_list[i]->length;
                    memcpy(tempBuf, payload.buffer_list[i]->data, payloadLen);
                    tempBuf = tempBuf + payloadLen;
                    length = length + payloadLen;
                }
            }

            std::string someipMessage(command, length);
            if (command) {
                free(command);
                command = nullptr;
            }
            ne_someip_log_debug("someipE2ECheck start: someipMessage.size:[%d]", someipMessage.size());
            bool isEvent = false;
            if (ne_someip_message_type_notification == messageType) {
                isEvent = true;
            }
            bool ret = NESomeIPE2EProtect::instance().someipE2ECheck(serviceId, instanceId, eventOrMethodId, &someipMessage,
                isEvent, retCheck, dataId, counter);
            if (!ret || someipMessage.size() > header.message_length) {
                ne_someip_log_error("someipE2ECheck error");
                return ne_someip_error_code_failed;
            }
            ne_someip_log_debug("someipE2ECheck end: someipMessage.size:[%d]  retCheck:[%d], dataId:[%d], counter:[%d]",
                someipMessage.size(), retCheck->IsOK(), dataId, counter);

            // check offset for someip
            ne_someip_client_id_t tempClientId;
            ne_someip_session_id_t tempSessionId;
            ne_someip_protocol_version_t tempProtocolVersion;
            ne_someip_interface_version_t tempInterfaceVersion;
            ne_someip_message_type_t tempMessageType;
            ne_someip_return_code_t tempReturnCode;
            uint32_t pos = 0;
            memcpy(&tempClientId, someipMessage.c_str() + pos, sizeof(tempClientId));
            pos += sizeof(tempClientId);
            memcpy(&tempSessionId, someipMessage.c_str() + pos, sizeof(tempSessionId));
            pos += sizeof(tempSessionId);
            memcpy(&tempProtocolVersion, someipMessage.c_str() + pos, sizeof(tempProtocolVersion));
            pos += sizeof(tempProtocolVersion);
            memcpy(&tempInterfaceVersion, someipMessage.c_str() + pos, sizeof(tempInterfaceVersion));
            pos += sizeof(tempInterfaceVersion);
            memcpy(&tempMessageType, someipMessage.c_str() + pos, sizeof(tempMessageType));
            pos += sizeof(tempMessageType);
            memcpy(&tempReturnCode, someipMessage.c_str() + pos, sizeof(tempReturnCode));
            pos += sizeof(tempReturnCode);
            ne_someip_log_debug("someipE2ECheck end: clientId:[%d], sessionId:[%d], protocolVersion:[%d], interfaceVersion:[%d], \
                messageType:[%d], returnCode:[%d], protectPayloadLength:[%d]",
                tempClientId, tempSessionId, tempProtocolVersion, tempInterfaceVersion, tempMessageType, tempReturnCode, (someipMessage.size() - pos));

            if (tempClientId != clientId ||
                tempSessionId != sessionId ||
                tempProtocolVersion != protocolVersion ||
                tempInterfaceVersion != interfaceVersion ||
                tempMessageType != messageType ||
                tempReturnCode != returnCode) {
                ne_someip_log_error("E2E profile offset error");
                return ne_someip_error_code_failed;
            }

            // TODO 这里默认payload.num为1
            if (payload.buffer_list[0]->data) {
                free(payload.buffer_list[0]->data);
            }
            uint32_t dataLengthChecked = someipMessage.size();
            payload.buffer_list[0]->length = dataLengthChecked - pos;
            header.message_length = dataLengthChecked;

            payload.buffer_list[0]->data = (uint8_t*)malloc(dataLengthChecked - pos);
            memcpy(payload.buffer_list[0]->data, someipMessage.c_str() + pos, dataLengthChecked - pos);
            return ne_someip_error_code_ok;
        } else {
            ne_someip_log_debug("serviceId[%d] instanceId[%d] eventOrMethodId[%d] no E2E Protect, need not Check",
                serviceId, instanceId, eventOrMethodId);
            return ne_someip_error_code_no_e2e_protect;
        }
    }

    ne_someip_error_code_t
    NESomeIPE2E::handleCheckStatus(ne_someip_service_id_t serviceId,
            ne_someip_instance_id_t instanceId,
            ne_someip_method_id_t eventId,
            E2E_state_machine::E2ECheckStatus checkStatus,
            std::shared_ptr<e2e::Result>& retCheck)
    {
        *retCheck = NESomeIPE2EProtect::instance().handleCheckStatus(serviceId, instanceId, eventId, checkStatus);
        return ne_someip_error_code_ok;
    }

/* EOF */
