
#ifndef DIAG_AGENT_DEF_H
#define DIAG_AGENT_DEF_H

#include <stdint.h>
#include <vector>
#include <string>

namespace hozon {
namespace netaos {
namespace diag {
namespace diag_agent {

// Service Identifier
enum DiagAgentServiceRequestOpc {
    DIAG_AGENT_SERVICE_REQUEST_OPC_BASE                    = 0X00,
    DIAG_AGENT_SERVICE_REQUEST_OPC_SESSION_CONTROL         = 0x10,
    DIAG_AGENT_SERVICE_REQUEST_OPC_ECU_RESET               = 0x11,
    DIAG_AGENT_SERVICE_REQUEST_OPC_DTC_CLEAR               = 0x14,
    DIAG_AGENT_SERVICE_REQUEST_OPC_READ_DTC_INFO           = 0x19,
    DIAG_AGENT_SERVICE_REQUEST_OPC_READ_DATA_IDENTIFIER    = 0x22,
    DIAG_AGENT_SERVICE_REQUEST_OPC_SECURITY_ACCESS         = 0x27,
    DIAG_AGENT_SERVICE_REQUEST_OPC_COMMUNICATION_CONTROL   = 0x28,
    DIAG_AGENT_SERVICE_REQUEST_OPC_WRITE_DATA_IDENTIFIER   = 0x2E,
    DIAG_AGENT_SERVICE_REQUEST_OPC_INPUT_OUTPUT_CONTROL    = 0x2F,
    DIAG_AGENT_SERVICE_REQUEST_OPC_ROUTINE_CONTROL         = 0x31,
    DIAG_AGENT_SERVICE_REQUEST_OPC_REQUEST_DOWNLOAD        = 0x34,
    DIAG_AGENT_SERVICE_REQUEST_OPC_TRANSFER_DATA           = 0x36,
    DIAG_AGENT_SERVICE_REQUEST_OPC_REQUEST_TRANSFER_EXIT   = 0x37,
    DIAG_AGENT_SERVICE_REQUEST_OPC_REQUEST_FILE_TRANSFER   = 0x38,
    DIAG_AGENT_SERVICE_REQUEST_OPC_TESTER_PRESENT          = 0x3E,
    DIAG_AGENT_SERVICE_REQUEST_OPC_CONTROL_DTC_SET         = 0x85,

    DIAG_AGENT_SERVICE_REQUEST_OPC_MAX_LEN
};

// Negetive Responce Code
enum DiagAgentNrcErrc {
    kNegativeHead = 0x7F,

    kGeneralReject = 0x10,
    kServiceNotSupported = 0x11,
    kSubfunctionNotSupported = 0x12,
    kIncorrectMessageLengthOrInvalidFormat = 0x13,
    kResponseTooLong = 0x14,
    kBusyRepeatRequest = 0x21,
    kConditionsNotCorrect = 0x22,
    kRequestSequenceError = 0x24,
    kNoResponseFromSubnetComponent = 0x25,
    kFailurePreventsExecutionOfRequestedAction = 0x26,
    kRequestOutOfRange = 0x31,
    kSecurityAccessDenied = 0x33,
    kInvalidKey = 0x35,
    kExceedNumberOfAttempts = 0x36,
    kRequiredTimeDelayNotExpired = 0x37,
    kUploadDownloadNotAccepted = 0x70,
    kTransferDataSuspended = 0x71,
    kGeneralProgrammingFailure = 0x72,
    kWrongBlockSequenceCounter = 0x73,
    kRequestCorrectlyReceivedResponsePending = 0x78,
    kSubFunctionNotSupportedInActiveSession = 0x7E,
    kServiceNotSupportedInActiveSession = 0x7F,
    kNotSupportSuppressPosMsgindication = 0xF0
};

// Init Result Code
enum DiagAgentInitResultCode {
    kSuccess = 0,
    kHandlerImplNull = -1,
    kAllInstanceNull = -2,
    kLoadConfigError = -3
};

struct DiagAgentDidDataInfo {
    uint16_t id;                                      // Data identifier value
    uint16_t dataLen;                                 // Data length(Each DID must have a fixed data length)
};

struct DiagAgentRidSubFuncInfo {
    uint8_t id;                                       // Routine identifier subfunction value(start: 0x01, stop: 0x02, result: 0x03)
    uint16_t reqLen;                                  // Request data length(Each RID subfunction must have a fixed request data length)
    uint16_t resLen;                                  // Reponse data length(Each RID subfunction must have a fixed response data length)
};

struct DiagAgentRidDataInfo {
    uint16_t id;                                      // Routine identifier value
    bool isSupportMultiStart;                         // Routine identifier is support multiple start or not(Must support without stop(0x02) subfunction)
    std::vector<DiagAgentRidSubFuncInfo> subFunc;     // List of supported subfunctions(Start(0x01) must be supported, stop(0x02) and result(0x03) are optional)
};


}  // namespace diag_agent
}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_AGENT_DEF_H
