
#ifndef UPDATE_CONFIG_MANAGER_H
#define UPDATE_CONFIG_MANAGER_H

#include "update_manager/config/hz_xml.h"
#include "update_manager/taskbase/task_object_def.h"
#include "update_manager/download/package_manager.h"
#include "update_manager/config/update_settings.h"

#include <string>
#include <unordered_map>
#include <vector>
#include <queue>

namespace hozon {
namespace netaos {
namespace update {


struct Partition {
    std::string name;
    std::string imgFile;
    std::string hash;
    std::string partitionA;
    std::string partitionB;
    std::string mountPoint;
    uint32_t offset;
};

typedef struct UpdateConfig {
    std::string version;
    std::string partNumber;
    std::string supplierCode;
    bool sameVersionCheck;
    std::vector<struct Partition> partition;
} UpdateConfig_t;

typedef struct {
    /// for transfer data
    uint8_t transType;          // 0: none, 1: TransData, 2: SecurityAccess, 3: TransFile
    std::string updateStep;     // "": for record log
    uint8_t addrType;           // 0: none, 1: physical, 2: functional
    std::vector<uint8_t> transData; // transfer data
    uint32_t transDataSize;     // 0: no check, other: may need add transData
    std::vector<uint8_t> recvExpect; // expect recv data
    uint32_t recvDataSize;      // 0: no check
    uint32_t waitTime;          // wait data transfer completed.
    uint32_t delayTime;         // after data transferred need bo delay
    /// for security access
    uint8_t securityLevel;      // 0: none, 1: applevel, 2: boot level
    uint32_t setcurityMask;
    /// for transfer file
    std::string filePath;       // "": invalid or not used
    uint8_t fileType;           // 0: none, 1: boot firmware, 2: app firmware, 3: cal firmware
    uint32_t memoryAddr;
    uint32_t memorySize;
    uint8_t beginProgress;
    uint8_t endProgress;
} UpdateCase_t;

typedef struct SensorEntityInfo {
    std::string partNumber;
    std::string supplierCode;
    std::string targetVersion;
    std::string flashDriverFirmwareName;
    std::string appFirmwareName;
    std::string calFirmwareName;
    std::string processFileName;
} SensorEntityInfo_t;

typedef struct SensorInfo {
    std::string name;           // target name
    uint8_t  updateType;        // 0: none, 1: docan, 2: doip, 3: update interface, 4: someip
    uint16_t canidTx;           // only as docan update, ecu canid tx
    uint16_t canidRx;           // only as docan update, ecu canid rx
    std::string ip;             // only as doip update, ecu ip v4 addr, just not support ipv6
    uint16_t logicalAddr;       // the target logical addr
    uint16_t functionAddr;      // the target function addr
    bool havaFileSystem;        // ecu whether has file system
    bool sameVersionCheck;      // ecu whether need do same version check
    uint8_t updateSequence;
    float progressWeight;
    std::vector<SensorEntityInfo_t> entitys;
    std::string processVersion;
} SensorInfo_t;

typedef struct Sensor {
    SensorInfo_t sensorInfo;
    std::vector<UpdateCase_t> process;
} Sensor_t;

typedef struct SensorManifest {
    std::string version;
    std::string processProportion;
    std::vector<Sensor_t> sensors;
} SensorManifest_t;

typedef struct SocInfo {
    std::string name;           // target name
    uint8_t  updateType;        // 0: none, 1: docan, 2: doip, 3: update interface, 4: someip
    uint16_t logicalAddr;       // the target logical addr
    bool havaFileSystem;        // ecu whether has file system
    bool sameVersionCheck;      // ecu whether need do same version check
    std::string partNumber;
    std::string supplierCode;
    std::string targetVersion;
    std::string firmwareName;
} SocInfo_t;

typedef struct SoC {
    SocInfo_t socInfo;
    // update case soc current not need.
} SoC_t;

typedef struct SocManifest {
    std::string version;
    std::string processProportion;
    std::vector<SoC_t> socs;
} SocManifest_t;

typedef struct VersionInfo {
    std::string majorVersion;
    std::string socVersion;
    std::string mcuVersion;
    std::string dsvVersion;
} VersionInfo_t;

class ConfigManager {
public:
    static ConfigManager &Instance()
    {
        static ConfigManager instance;
        return instance;
    }

    int32_t Init();
    int32_t Start();
    int32_t Stop();
    int32_t Deinit();

    bool IsSensorUpdate();
    bool IsSocUpdate();

    bool ClearSensorUpdate();
    bool ClearSocUpdate();
    
    int32_t ParseUpdateFileList();
    int32_t ParseSensorManifest();
    int32_t ParseSocManifest();
    bool ParseAllConfig();

    bool GetVin(std::vector<uint8_t>& vin);
    bool GetDate(std::vector<uint8_t>& date);
    bool GetTesterSN(std::vector<uint8_t>& sn);

    std::unordered_map<std::string, std::string>& GetFileList();
    UpdateConfig_t& GetUpdateConfig();
    SensorManifest_t& GetSensorManifest();
    bool GetSensorManifestByName(const std::string& sensorName, Sensor_t& info);
    SocManifest_t& GetSocManifest();

    bool GetMajorVersion(std::string& majorVersion);
    bool GetSocVersion(std::string& socVersion);
    bool GetMcuVersion(std::string& mcuVersion);
    bool GetDsvVersion(std::string& dsvVersion);

    bool MountSensors();
    bool UmountSensors();
    bool UmountAndRemoveSensors();
    bool Ecb_decryptFile(const std::string& inputFilePath, const std::string& outputFilePath, const std::string& key);
private:
    ConfigManager();
    ConfigManager(const ConfigManager &);
    ConfigManager & operator = (const ConfigManager &);

    std::unordered_map<std::string, std::string> update_filelist_map_;
    UpdateConfig_t      update_config_;
    SensorManifest_t    sensor_manifest_;
    SocManifest_t       soc_manifest_;
    VersionInfo_t       version_info_;
};

}  // namespace update
}  // namespace netaos
}  // namespace hozon
#endif  // UPDATE_CONFIG_MANAGER_H