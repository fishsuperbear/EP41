#pragma once

#include <stdint.h>
#include <mutex>
#include <algorithm>
#include <regex>
#include "update_manager/common/data_def.h"
#include "cfg/include/config_param.h"

using namespace hozon::netaos::cfg;

namespace hozon {
namespace netaos {
namespace update {

class OTAStore {
public:

    static OTAStore* Instance();

    void Init();
    void Deinit();

    // F188
    bool ReadECUSWData(std::vector<uint8_t>& ecuSWData);
    bool WriteECUSWData(const std::vector<uint8_t>& ecuSWData);

    // F1C0 (Major Version)
    bool ReadECUVersionData(std::vector<uint8_t>& ecuVersionData);
    bool WriteECUVersionData(const std::vector<uint8_t>& ecuVersionData);

    // F198 
    bool WriteTesterSNData(const std::vector<uint8_t>& testerSNData);

    // F199
    bool WriteProgrammingDateData(const std::vector<uint8_t>& programmingDateData);

    // SOC version
    bool ReadSocVersionData(std::string& socVersion);
    bool WriteSocVersionData(const std::string& socVersion);

    // DSV version
    bool ReadDsvVersionData(std::string& dsvVersion);
    bool WriteDsvVersionData(const std::string& dsvVersion);

    // MCU version
    bool ReadMcuVersionData(std::string& mcuVersion);
    bool WriteMcuVersionData(const std::string& mcuVersion);

    // switch version
    bool ReadSwitchVersionData(std::string& switchVersion);
    bool WriteSwitchVersionData(const std::string& switchVersion);

    // PackageName
    bool ReadPackageNameData(std::string& pkgName);
    bool WritePackageNameData(const std::string& pkgName);

    // Sensor version
    bool WriteSensorVersionData(const std::string& sensorName, const std::string& sensorVersion);
    bool ReadSensorVersionData(const std::string& sensorName, std::string& sensorVersion);

    // Dynamic version
    bool WriteDynamicSensorVersionData(const std::string& sensorName, const std::string& sensorVersion);

    // cmd upgrade flag
    bool ReadCmdFlagData(bool& cmdFlag);
    bool WriteCmdFlagData(const bool& cmdFlag);

private:
    enum CheckType {
        kNumber = 0x00,
        kLetter = 0x01,
        kNumberAndLetter = 0x02,
        kNumberAndLetterAndSymbol = 0x03
    };
    static bool DataCheck(const CheckType& type, const std::vector<uint8_t>& data);
    void ReadDataFromCfg(const std::string& dataKey, const UpdateManagerInfoDataType& dataType, std::vector<uint8_t>& data);
    static std::vector<std::string> Split(const std::string& inputStr, const std::string& regexStr = " ");

private:
    OTAStore();
    ~OTAStore();
    OTAStore(const OTAStore &);
    OTAStore & operator = (const OTAStore &);

    static std::mutex m_mtx;
    static OTAStore* m_pInstance;

    ConfigParam* cfg_mgr_;
};

}  // namespace update
}  // namespace netaos
}  // namespace hozon
