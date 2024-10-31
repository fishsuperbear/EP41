#ifndef SYSTEM_MONITOR_DEF_H
#define SYSTEM_MONITOR_DEF_H

#include <stdint.h>
#include <vector>
#include <string>
#include <unordered_map>

namespace hozon {
namespace netaos {
namespace system_monitor {

const std::string CONTROL_OUTPUT_LINE = "---------------------------------------------------------------------------------------------------\n";

enum SystemMonitorSubFunctionId {
    kCpuMonitor = 0x00,
    kMemMonitor = 0x01,
    kDiskMonitor = 0x02,
    kEmmcMonitor = 0x03,
    kTemperatureMonitor = 0x04,
    kVoltageMonitor = 0x05,
    kFileSystemMonitor = 0x06,
    kProcessMonitor = 0x07,
    kNetworkMonitor = 0x08,

    kAllMonitor = 0x09
};

enum SystemMonitorSubFunctionControlType {
    kMonitorSwitch = 0x00,
    kMonitorCycle = 0x01,
    kRecordFileCycle = 0x02,
    kRecordFilePath = 0x03,
    kIsAlarm = 0x04,
    kAlarmValue = 0x05,
    kPostProcessingSwitch = 0x06
};

struct SystemMonitorSubFunctionInfo {
    std::string shortName;
    SystemMonitorSubFunctionId id;
    std::string monitorSwitch;
    uint32_t monitorCycle;
    uint32_t recordFileCycle;
    std::string recordFilePath;
    bool isAlarm;
    uint8_t alarmValue;
    std::string postProcessingSwitch;
};

struct SystemMonitorConfigInfo {
    std::string LogAppName;
    std::string LogAppDescription;
    std::string LogContextName;
    uint8_t LogLevel;
    uint8_t LogMode;
    std::string LogFilePath;
    uint32_t MaxLogFileNum;
    uint32_t MaxSizeOfLogFile;
    std::string DebugSwitch;
    std::unordered_map<SystemMonitorSubFunctionId, SystemMonitorSubFunctionInfo> subFunction;
};

struct SystemMonitorTcpConfigInfo {
    std::string TcpEthName;
    uint32_t TcpPort;
    uint8_t TcpMaxClients;
};

struct SystemMonitorControlEventInfo {
    SystemMonitorSubFunctionId id;
    SystemMonitorSubFunctionControlType type;
    std::string value;
};

struct SystemMonitorSendFaultInfo {
    uint32_t faultId;
    uint8_t faultObj;
    uint8_t faultStatus;
};

struct SystemMonitorFileMonitorInfo {
    std::string pathName;
    bool isRecursive;
};

struct SystemMonitorDiskMonitorListInfo {
    std::string pathName;
    bool isTraverseSubdir;
};

struct SystemMonitorDiskMonitorPartitionInfo {
    std::string partition;
    uint32_t reservedSize;
    uint8_t partitionAlarmValue;
    bool isIncludeSpecialPostProcessing;
    bool isDeleteEmptyDir;
    std::vector<SystemMonitorDiskMonitorListInfo> deleteFilesByWildcardList;
    std::vector<SystemMonitorDiskMonitorListInfo> deleteFilesByPathList;
};

enum SystemMonitorDiskMonitorLogMoveType {
    kSoc,
    kMcu
};

enum SystemMonitorDiskMonitorGetFilesType {
    kFixedPrefix,
    kSizeExceedsLimit
};

struct SystemMonitorDiskMonitorLogMoveListInfo {
    std::string filePrefix;
    uint8_t reservedFileNum;
};

struct SystemMonitorDiskMonitorGetFilesTypeInfo {
    SystemMonitorDiskMonitorGetFilesType type;
    std::string value;
};

}  // namespace system_monitor
}  // namespace netaos
}  // namespace hozon
#endif  // SYSTEM_MONITOR_DEF_H
