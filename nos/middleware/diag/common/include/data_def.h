#ifndef DATA_DEF_H
#define DATA_DEF_H

#include <stdint.h>
#include <vector>
#include <list>
#include <string>

namespace hozon {
namespace netaos {
namespace diag {

const std::string DiagSession_Default = "DefaultSession";
const std::string DiagSession_Programming = "ProgrammingSession";
const std::string DiagSession_Extend = "ExtendSession";

const std::string DiagAccess_All = "AllSessionAccessPermission";
const std::string DiagAccess_Extended = "OnlyExtendedSessionAccessPermission";
const std::string DiagAccess_Programming = "OnlyProgrammingSessionAccessPermission";
const std::string DiagAccess_NonDefault = "NonDefaultSessionAccessPermission";
const std::string DiagAccess_NonProgramming = "NonProgrammingSessionAccessPermission";
const std::string DiagAccess_NonDefault_Level1_LevelFBL = "NonDefaultSessionWithLevel1AndLevelFBLAccessPermission";
const std::string DiagAccess_Programming_LevelFBL = "OnlyProgrammingSessionWithLevelFBLAccessPermission";
const std::string DiagAccess_Extended_Level1 = "OnlyExtendedSessionWithLevel1AccessPermission";
const std::string DiagAccess_Extended_LevelTEST = "OnlyExtendedSessionWithLevelTESTAccessPermission";

const std::string DiagSecurityLevel_Non = "LevelNon";
const std::string DiagSecurityLevel_Level1 = "Level1";
const std::string DiagSecurityLevel_LevelTEST = "LevelTEST";
const std::string DiagSecurityLevel_LevelFBL = "LevelFBL";

enum ReadJsonFileReturnCode {
    RET_OK,
    RET_ERROR
};

enum DiagSessionId {
    DiagSessionId_Default                          = 0x01,
    DiagSessionId_Programming                      = 0x02,
    DiagSessionId_Extend                           = 0x03
};

enum DiagAccessId {
    DiagAccessId_All                               = 0x01,
    DiagAccessId_Extended                          = 0x02,
    DiagAccessId_Programming                       = 0x03,
    DiagAccessId_NonDefault                        = 0x04,
    DiagAccessId_NonProgramming                    = 0x05,
    DiagAccessId_NonDefault_Level1_LevelFBL        = 0x06,
    DiagAccessId_Programming_LevelFBL              = 0x07,
    DiagAccessId_Extended_Level1                   = 0x08,
    DiagAccessId_Extended_LevelTEST                = 0x09
};

enum DiagSecurityLevelId {
    DiagSecurityLevelId_Non                        = 0x00,
    DiagSecurityLevelId_Level1                     = 0x03,
    DiagSecurityLevelId_LevelTEST                  = 0x05,
    DiagSecurityLevelId_LevelFBL                   = 0x11
};

struct DiagConfigInfo {
    std::string LogAppName;
    std::string LogAppDescription;
    std::string LogContextName;
    std::uint8_t LogLevel;
    std::uint8_t LogMode;
    std::string LogFilePath;
    std::uint32_t MaxLogFileNum;
    std::uint32_t MaxSizeOfLogFile;
    std::string DebugSwitch;
};

struct DiagSoftWareClusterDataInfo {
    std::string shortName;
    uint8_t id;
    std::vector<uint16_t> functionAddressList;
    uint16_t physicAddress;
    std::vector<uint16_t> remoteAddressList;
    uint16_t updateManagerAddress;
    std::vector<std::string> tpInstances;
};

struct DiagTransferConfigDataInfo {
    std::string shortName;
    uint8_t id;
    uint64_t transferSize;
    std::vector<std::string> pathWhiteList;
};

struct DiagExternalServiceConfigDataInfo {
    std::string shortName;
    uint8_t id;
    std::vector<uint8_t> supportSid;
    std::vector<uint16_t> supportReadDid;
    std::vector<uint16_t> supportWriteDid;
    std::vector<uint16_t> supportRid;
};

struct DiagSessionDataInfo {
    std::string shortName;
    uint8_t id;
    uint16_t p2Server;
    uint16_t p2ServerStar;
    uint16_t s3;
};

struct DiagSecurityLevelDataInfo {
    std::string shortName;
    uint8_t id;
    uint32_t mask;
    uint16_t seedSize;
    uint16_t keySize;
    uint16_t numFailedSecurityAccess;
    uint16_t securityDelayTime;
};

struct DiagAccessPermissionDataInfo {
    std::string shortName;
    uint8_t id;
    std::vector<uint8_t> allowedSessions;
    std::vector<uint8_t> allowedSecurityLevels;
};

struct DiagSubFuncDataInfo {
    std::string shortName;
    uint8_t id;
    bool isSupportFunctionAddr;
    bool isSupportSuppressPosMsgindication;
    uint8_t accessPermission;
};

struct DiagRidSubFuncDataInfo {
    std::string shortName;
    uint8_t id;
    uint8_t accessPermission;
    uint32_t requestLen;
    uint32_t replyLen;
};

struct DiagSidDataInfo {
    std::string shortName;
    uint8_t id;
    bool isSupportFunctionAddr;
    uint8_t accessPermission;
    uint16_t maxPendingNumber;
    std::vector<DiagSubFuncDataInfo> subFunctions;
};

struct DiagDidDataInfo {
    std::string shortName;
    uint16_t id;
    uint8_t readAccessPermission;
    bool isSupportWrite;
    uint8_t writeAccessPermission;
    uint16_t dataSize;
};

struct DiagRidDataInfo {
    std::string shortName;
    uint16_t id;
    uint8_t accessPermission;
    bool isSupportMultiStart;
    std::vector<DiagRidSubFuncDataInfo> ridSubFunctions;
};

struct DiagDtcDataInfo {
    std::string dtc;
    uint32_t dtcValue;
    uint32_t clusterId;
    uint32_t faultKey;
    uint8_t notifyHmi;
    uint8_t agingAllowed;
    uint8_t agingCount;
    uint8_t tripCount;
    std::vector<uint16_t> extendDids;
};

struct DiagDemDataInfo {
    uint8_t extendDataRecordNo;
    uint8_t snapshotRecordNo;
    uint8_t snapshotCount;
    uint8_t snapshotIdentifiers;
    uint8_t extendIdentifiers;
    std::vector<uint16_t> snapshotDids;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DATA_DEF_H
