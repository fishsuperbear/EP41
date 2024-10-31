
#ifndef DIAG_SERVER_DEF_H
#define DIAG_SERVER_DEF_H

#include <stdint.h>
#include <vector>
#include <string>

namespace hozon {
namespace netaos {
namespace diag {

const uint8_t DIAG_SERVER_READ_DID_MAX_NUMBER = 2;
const uint8_t DIAG_SERVER_SUPPRESS_POS_RES_MSG_INDICATION_BIT = 0x80;

enum DiagServerServiceRequestOpc {
    DIAG_SERVER_SERVICE_REQUEST_OPC_BASE                    = 0X00,
    DIAG_SERVER_SERVICE_REQUEST_OPC_SESSION_CONTROL         = 0x10,
    DIAG_SERVER_SERVICE_REQUEST_OPC_ECU_RESET               = 0x11,
    DIAG_SERVER_SERVICE_REQUEST_OPC_DTC_CLEAR               = 0x14,
    DIAG_SERVER_SERVICE_REQUEST_OPC_READ_DTC_INFO           = 0x19,
    DIAG_SERVER_SERVICE_REQUEST_OPC_READ_DATA_IDENTIFIER    = 0x22,
    DIAG_SERVER_SERVICE_REQUEST_OPC_SECURITY_ACCESS         = 0x27,
    DIAG_SERVER_SERVICE_REQUEST_OPC_COMMUNICATION_CONTROL   = 0x28,
    DIAG_SERVER_SERVICE_REQUEST_OPC_WRITE_DATA_IDENTIFIER   = 0x2E,
    DIAG_SERVER_SERVICE_REQUEST_OPC_INPUT_OUTPUT_CONTROL    = 0x2F,
    DIAG_SERVER_SERVICE_REQUEST_OPC_ROUTINE_CONTROL         = 0x31,
    DIAG_SERVER_SERVICE_REQUEST_OPC_REQUEST_DOWNLOAD        = 0x34,
    DIAG_SERVER_SERVICE_REQUEST_OPC_TRANSFER_DATA           = 0x36,
    DIAG_SERVER_SERVICE_REQUEST_OPC_REQUEST_TRANSFER_EXIT   = 0x37,
    DIAG_SERVER_SERVICE_REQUEST_OPC_REQUEST_FILE_TRANSFER   = 0x38,
    DIAG_SERVER_SERVICE_REQUEST_OPC_TESTER_PRESENT          = 0x3E,
    DIAG_SERVER_SERVICE_REQUEST_OPC_CONTROL_DTC_SET         = 0x85,

    DIAG_SERVER_SERVICE_REQUEST_OPC_MAX_LEN
};

enum DiagServerServiceReplyOpc {
    DIAG_SERVER_SERVICE_REPLY_OPC_BASE                    = 0X40,
    DIAG_SERVER_SERVICE_REPLY_OPC_SESSION_CONTROL         = 0x50,
    DIAG_SERVER_SERVICE_REPLY_OPC_ECU_RESET               = 0x51,
    DIAG_SERVER_SERVICE_REPLY_OPC_DTC_CLEAR               = 0x54,
    DIAG_SERVER_SERVICE_REPLY_OPC_READ_DTC_INFO           = 0x59,
    DIAG_SERVER_SERVICE_REPLY_OPC_READ_DATA_IDENTIFIER    = 0x62,
    DIAG_SERVER_SERVICE_REPLY_OPC_SECURITY_ACCESS         = 0x67,
    DIAG_SERVER_SERVICE_REPLY_OPC_COMMUNICATION_CONTROL   = 0x68,
    DIAG_SERVER_SERVICE_REPLY_OPC_WRITE_DATA_IDENTIFIER   = 0x6E,
    DIAG_SERVER_SERVICE_REPLY_OPC_INPUT_OUTPUT_CONTROL    = 0x6F,
    DIAG_SERVER_SERVICE_REPLY_OPC_ROUTINE_CONTROL         = 0x71,
    DIAG_SERVER_SERVICE_REPLY_OPC_REQUEST_DOWNLOAD        = 0x74,
    DIAG_SERVER_SERVICE_REPLY_OPC_TRANSFER_DATA           = 0x76,
    DIAG_SERVER_SERVICE_REPLY_OPC_REQUEST_TRANSFER_EXIT   = 0x77,
    DIAG_SERVER_SERVICE_REPLY_OPC_REQUEST_FILE_TRANSFER   = 0x78,
    DIAG_SERVER_SERVICE_REPLY_OPC_TESTER_PRESENT          = 0x7E,
    DIAG_SERVER_SERVICE_REPLY_OPC_CONTROL_DTC_SET         = 0xC5,

    DIAG_SERVER_SERVICE_REPLY_OPC_MAX_LEN
};

enum DiagServerNrcErrc {
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

enum DiagServerSessionCode {
    kDefaultSession = 0x01,
    kProgrammingSession = 0x02,
    kExtendedSession = 0x03
};

enum DiagUdsBusType {
    kDocan = 0x01,
    kDoip = 0x02,
    kServer = 0x03,
    kDoip_DisConn = 0x20
};

enum DiagTargetAddressType {
    kPhysical = 0x00,
    kFunctional = 0x01
};

enum DoipNetlinkStatus {
    kUp = 0x00,
    kDown = 0x01
};

struct DiagServerUdsMessage {
    uint32_t id;
    uint8_t suppressPosRspMsgIndBit;
    uint8_t pendingRsp;
    uint16_t udsSa;
    uint16_t udsTa;
    DiagTargetAddressType taType;
    std::vector<uint8_t> udsData;

    void Copy(const DiagServerUdsMessage& msg)
    {
        id = msg.id;
        suppressPosRspMsgIndBit = msg.suppressPosRspMsgIndBit;
        pendingRsp = msg.pendingRsp;
        taType = msg.taType;
        udsSa = msg.udsSa;
        udsTa = msg.udsTa;
        // udsData = msg.udsData;
    }

    // change udsSa and udsTa
    void Change(const DiagServerUdsMessage& msg)
    {
        id = msg.id;
        suppressPosRspMsgIndBit = msg.suppressPosRspMsgIndBit;
        pendingRsp = msg.pendingRsp;
        udsSa = msg.udsTa;
        udsTa = msg.udsSa;
        taType = msg.taType;
        // udsData = msg.udsData;
    }
};

struct DiagServerReqUdsMessage {
    uint16_t udsSa;
    uint16_t udsTa;
    DiagUdsBusType busType;
    DiagTargetAddressType taType;
    std::vector<uint8_t> udsData;
};

struct DiagServerRespUdsMessage {
    uint16_t udsSa;
    uint16_t udsTa;
    uint32_t result;
    DiagUdsBusType busType;
    DiagTargetAddressType taType;
    std::vector<uint8_t> udsData;
};

struct DiagServerChassisData {
    uint8_t gearDisplay;
    float outsideTemp;
    float odometer;
    uint8_t powerMode;
    uint8_t igStatus;
    bool vehicleSpeedValid;
    double vehicleSpeed;
};

union DiagDate
{
    struct
    {
        int16_t wYear;
        int16_t wMonth;
        int16_t wDay;
    } Date;

    int64_t iDate;
};

union DiagTime
{
    struct
    {
        int16_t wHour;
        int16_t wMinute;
        int16_t wSecond;
    } Time;

    int64_t iTime;
};

struct DiagSnapshotData
{
    uint64_t date;
    uint8_t voltage;
    uint32_t odometer;
    uint16_t speed;
    uint8_t igStatus;
};

struct DiagDidData
{
    uint16_t did;
    std::vector<uint8_t> didData;
};

struct DiagDtcData
{
    uint32_t dtc;
    uint8_t dtcStatus; // 0 is recover, 1 is occur
    uint32_t snapshotId;
    DiagSnapshotData stSnapshotData;
    std::vector<DiagDidData> vSnapshotData;
    std::vector<DiagDidData> vExtendData;
};

enum class DIAG_CONTROLDTCSTATUSTYPE : uint8_t
{
    kDTCSettingOn = 0x01,
    kDTCSettingOff = 0x02
};

enum class Doip_Req_Channel : uint16_t
{
    kDefault    = 0x00,
    kSomeip     = 0x01,
    kNotSomeip  = 0x02,
};

enum class Docan_Req_Channel : uint16_t
{
    kDefault    = 0x00,
    kSomeip     = 0x01,
    kNotSomeip  = 0x02,
};

enum class Server_Req_Channel : uint16_t
{
    kDefault    = 0x00,
    kSomeip     = 0x01,
    kNotSomeip  = 0x02,
};

enum class Req_Channel : uint16_t
{
    kServer   = 0x00,
    kDoip     = 0x01,
    kDocan    = 0x02,
};

enum DiagServerInfoDataType {
    kNumber = 0x00,
    kLetter = 0x01,
    kNumberAndLetter = 0x02,
    kNumberAndLetterAndSymbol = 0x03,
    kHEX = 0x04,
    kBCD = 0x05,
    kASCII = 0x06
};

enum DiagUpdateStatus {
    kDefault = 0x00,
    kUpdating = 0x01,
    kUpdated = 0x02
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_DEF_H
