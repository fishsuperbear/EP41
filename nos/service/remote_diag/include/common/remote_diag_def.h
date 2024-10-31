#ifndef REMOTE_DIAG_DEF_H
#define REMOTE_DIAG_DEF_H

#include <stdint.h>
#include <vector>
#include <string>

namespace hozon {
namespace netaos {
namespace remote_diag {

const std::string REGEX = " ";
const std::vector<std::string> REMOTE_DIAG_REQUEST_TYPE = {"uds_command", "file_upload", "file_download", "plugin_run", "plugin_run_result", "switch_control", "query_dir_info"};
const std::vector<std::string> REMOTE_DIAG_PLUGIN_RUN_STATUS = {"Default", "Executing", "Terminated", "Timeout", "Failed", "Succeed", "Result"};
struct RemoteDiagConfigInfo {
    std::string LogAppName;
    std::string LogAppDescription;
    std::string LogContextName;
    std::uint8_t LogLevel;
    std::uint8_t LogMode;
    std::string LogFilePath;
    std::uint32_t MaxLogFileNum;
    std::uint32_t MaxSizeOfLogFile;
    std::string DebugSwitch;
    std::uint32_t FileTransferSize;
    std::string FileCompressDirPath;
    std::string FileDownloadDirPath;
    std::string RocketMQReqGroup;
    std::string RocketMQResGroup;
    std::string RocketMQAddress;
    std::string RocketMQReqTopic;
    std::string RocketMQResTopic;
    std::string RocketMQResTag;
    std::string RocketMQResKeys;
    std::string RocketMQDomain;
    std::string RocketMQAccessKey;
    std::string RocketMQSecretKey;
    std::uint16_t DiagServerAddress;
    std::vector<uint16_t> RemoteAddressList;
    std::vector<uint16_t> DoipAddressList;
    std::vector<uint16_t> DocanAddressList;
};

enum DiagUdsNrcErrc {
    kNegativeHead = 0x7F,

    kErrorSa = 0x0A,
    kErrorTa = 0x0B,
    kRequestBusy = 0x0C,
    kVehicleInMotion = 0x0D
};

enum DiagUdsBusType {
    kDocan = 0x01,
    kDoip = 0x02,
    kServer = 0x03
};

enum DiagTargetAddressType {
    kPhysical = 0x00,
    kFunctional = 0x01
};

enum RemoteDiagDataType {
    kUdsCommand = 0x00,
    kFileUpload = 0x01,
    kFileDownload = 0x02,
    kPluginRun = 0x03,
    kPluginRunResult = 0x04,
    kSwitchControl = 0x05,
    kQueryDirInfo = 0x06
};

enum RemoteDiagPluginRunStatusType {
    kDefault = 0x00,
    kExecuting = 0x01,
    kTerminated = 0x02,
    kTimeout = 0x03,
    kFailed = 0x04,
    kSucceed = 0x05,
    kResult = 0x06
};

struct RemoteDiagReqUdsMessage {
    uint16_t udsSa;
    uint16_t udsTa;
    DiagUdsBusType busType;
    std::vector<uint8_t> udsData;
};

struct RemoteDiagPluginRunInfo {
    std::string sa;
    std::string ta;
    std::string pluginPackageName;
};

struct RemoteDiagPluginRunResultInfo {
    std::string sa;
    std::string ta;
    std::string pluginName;
};

struct RemoteDiagPluginDescribeInfo {
    std::string pluginName;
    std::string pluginDescribe;
    std::string pluginRunParameters;
    bool isNeedCompressResult;
    std::string unCompressFile;
    uint16_t runTimeout;
    std::string pluginPath;
    std::string scriptPath;
    RemoteDiagPluginRunStatusType runStatus;
    std::string runResultPath;
};

struct RemoteDiagSwitchControlInfo {
    std::string sa;
    std::string ta;
    std::string switchName;
    std::string control;
};

struct RemoteDiagQueryDirInfo {
    std::string sa;
    std::string ta;
    std::string infoType;
    std::string dirFilePath;
};

struct RemoteDiagFileUploadInfo {
    std::string sa;
    std::string ta;
    std::string uploadFileType;
    std::string uploadFilePath;
};

struct RemoteDiagFileDownloadInfo {
    std::string sa;
    std::string ta;
    std::string downloadDirPath;
    std::string downloadFileName;
    std::string md5;
    uint32_t blockCount;
    uint32_t blockSize;
    std::string data;

};

enum REMOTE_DIAG_EXTENSION {
    REMOTE_DIAG_FILE_UPLOAD = 1,
    REMOTE_DIAG_FILE_DOWNLOAD = 2,
    REMOTE_DIAG_PLUGIN_RUN = 3,
    REMOTE_DIAG_GET_PLUGIN_RUN_RESULT = 4,
    REMOTE_DIAG_SWITCH_CONTROL = 5,

    REMOTE_DIAG_DEFAULT = 255
};

}  // namespace remote_diag
}  // namespace netaos
}  // namespace hozon
#endif  // REMOTE_DIAG_DEF_H
