
#ifndef PHM_SERVER_DEF_H
#define PHM_SERVER_DEF_H

#include <mutex>
#include <memory>
#include <vector>
#include <string>

namespace hozon {
namespace netaos {
namespace phm_server {

const uint64_t NORMAL_FAULT_LOCK_VALUE = 0x4000000000000000;

struct HzFaultEventToMCU
{
    uint16_t faultId;
    uint8_t faultObj;
    uint8_t faultStatus;
    uint8_t postProcessArray[60];
};

struct FaultStrategy {
    uint8_t notifyMcu;
    uint8_t notifyApp;
    uint8_t restartproc;
    uint8_t dtcMapping;
};

struct FaultAction {
    uint8_t record;
    uint8_t analysis;
    FaultStrategy strategy;
};

typedef struct FaultInfo {
    uint32_t faultId;
    uint8_t faultObj;
    uint16_t faultClusterId;
    uint16_t faultLevel;
    FaultAction faultAction;
    std::string faultProcess;
    std::string faultDscribe;
    std::string faultDomain;
    uint64_t faultOccurTime;
    uint8_t faultStatus;
    uint32_t faultRecordCount;
    uint32_t dtcCode;
} Fault_t;

struct AnalysisFault {
    std::string faultProcess;       // report fault process
    std::string faultDscribe;
    std::string faultDomain;
    uint32_t key;
    uint64_t faultOccurTime;        // fault occur time msec
    uint32_t matureCount;           // fault mature count
    uint32_t recoverCount;          // fault recover count
    uint32_t fatalCount;
    uint32_t criticalCount;
    uint32_t normalCount;
    uint32_t infoCount;

    uint64_t avgCyc;                // msec
    uint64_t minGap;                // msec
    uint64_t maxGap;                // msec
};

struct AnalysisFaultStatus {
    std::string faultProcess;       // report fault process
    std::string faultDomain;
    uint32_t key;
    uint8_t faultStatus;
    uint32_t count;
};

struct AnalisysNonstandard {
    std::string faultProcess;       // report fault process
    std::string faultDomain;
    uint32_t key;
    uint32_t count;
};

struct AnalisysOverCount {
    std::string faultProcess;       // report fault process
    std::string faultDscribe;
    std::string faultDomain;
    uint32_t key;
    uint32_t count;
};

struct SystemCheckFaultInfo {
    std::string faultDomain;
    uint64_t faultOccurTime;
    uint32_t faultId;
    uint8_t faultObj;
    uint8_t faultStatus;
    uint32_t faultOccurCount;
    uint32_t faultRecoverCount;
};

struct ProcInfo {
    std::string procName;
    uint32_t faultId;
    uint8_t faultObj;
    uint8_t retryCount;
};

struct PhmConfigInfo {
    std::string LogAppName;
    std::string LogAppDescription;
    std::string LogContextName;
    std::uint8_t LogLevel;
    std::uint8_t LogMode;
    std::string LogFilePath;
    std::uint32_t MaxLogFileNum;
    std::uint32_t MaxSizeOfLogFile;
    std::uint32_t CollectFaultFileSize;
    std::uint32_t CollectFaultMaxCnt;
    std::string DebugSwitch;
    std::string RestartProcSwitch;
    std::string AnalysisSwitch;
    std::uint32_t SystemCheckTime;
    std::uint32_t ProcCheckTime;
    std::uint32_t AnalysisTime;
    std::string LockFaultToHMISwitch;
    std::string LockFaultCurrentVersion;
};

struct Timestamp {
    uint32_t nsec;
    uint32_t sec;
};

struct SendFaultPack {
    std::string faultDomain;
    uint64_t faultOccurTime;
    uint32_t faultId;
    uint8_t faultObj;
    uint8_t faultStatus;
};

struct FaultClusterItem {
    std::string clusterName;
    uint8_t bitPosition;
};

enum ReadJsonFileReturnCode {
    OK,
    ERROR
};

struct FaultLockInfo {
    uint32_t faultId;               // Fault id
    uint32_t faultObj;              // Fault obj
    uint32_t lockCount;             // Conditions for the number of fault lock
    uint32_t recoverCount;          // Conditions for the number of fault recover
    std::string faultToHMIData;     // Data to HMI when fault not lock
    std::string lockFaultToHMIData; // Data to HMI when fault lock
    uint8_t isBlockedFault;         // Blocked Fault[1] Else[0]
    uint32_t faultCount;            // Fault occured number
    uint8_t isHandled;              // Current cycle handled[1] nothandled[0]
    uint32_t lockedNumber;          // Fault locked number
    uint32_t faultRecoverCount;     // Fault not occured number of consecutive
    uint8_t isNeedToRecover;        // Current cycle needToRecover[1] notNeedToRecover[0]

};

struct FaultLockReportInfo {
    uint32_t faultId;
    uint32_t faultObj;
    uint32_t faultStatus;
    bool isInhibitWindowFault;
};

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
#endif  // PHM_SERVER_DEF_H
