#include "update_manager/record/ota_store.h"
#include "update_manager/log/update_manager_logger.h"

namespace hozon {
namespace netaos {
namespace update {

OTAStore* OTAStore::m_pInstance = nullptr;
std::mutex OTAStore::m_mtx;
const uint8_t ECU_SW_DATA_LENGTH = 8;
const uint8_t ECU_VERSION_DATA_LENGTH = 8;
const uint8_t TESTER_SN_DATA_LENGTH = 10;
const uint8_t PROGRAMMING_DATE_DATA_LENGTH = 4;

OTAStore::OTAStore()
: cfg_mgr_(ConfigParam::Instance())
{
}

OTAStore::~OTAStore()
{
}

OTAStore*
OTAStore::Instance()
{
    if (nullptr == m_pInstance) {
        std::unique_lock<std::mutex> lck(m_mtx);
        if (nullptr == m_pInstance) {
            m_pInstance = new OTAStore();
        }
    }

    return m_pInstance;
}

void
OTAStore::Init()
{   UM_INFO << "OTAStore::Init.";
    if (nullptr != cfg_mgr_) {
        cfg_mgr_->Init(3000);
    }
    UM_INFO << "OTAStore::Init Done.";
}

void
OTAStore::Deinit()
{
    UM_INFO << "OTAStore::Deinit.";
    if (nullptr != cfg_mgr_) {
        cfg_mgr_->DeInit();
    }

    if (m_pInstance != nullptr) {
        delete m_pInstance;
        m_pInstance = nullptr;
    }
    UM_INFO << "OTAStore::Deinit Done.";
}

bool
OTAStore::ReadECUSWData(std::vector<uint8_t>& ecuSWData)
{
    UPDATE_LOG_I("OTAStore::ReadECUSWData");

    if (nullptr == cfg_mgr_) {
        UPDATE_LOG_E("OTAStore::ReadECUSWData cfg_mgr_ is nullptr.");
        return false;
    }

    std::vector<uint8_t> data;
    ReadDataFromCfg("dids/F188", UpdateManagerInfoDataType::kASCII, data);

    if (0 == data.size()) {
        UPDATE_LOG_W("OTAStore::ReadECUSWData get data is empty.");
        return false;
    }

    if (data.size() != ECU_SW_DATA_LENGTH) {
        UPDATE_LOG_E("OTAStore::ReadECUSWData ecu_sw_data_ size error. correct f188_data_.size: %u , cfg get data.size: %zu "
                    ,ECU_SW_DATA_LENGTH, data.size());
        return false;
    }

    for (auto& item : data) {
        ecuSWData.push_back(item);
    }

    return true;
}

bool
OTAStore::WriteECUSWData(const std::vector<uint8_t>& ecuSWData)
{
    UPDATE_LOG_I("OTAStore::WriteECUSWData");
    if (ecuSWData.size() != ECU_SW_DATA_LENGTH) {
        UPDATE_LOG_E("OTAStore::WriteECUSWData ecuSWData size error. need ecuSWData.size: %u, ecuSWData.size: %zu."
                    , ECU_SW_DATA_LENGTH, ecuSWData.size());
        return false;
    }
    if (nullptr == cfg_mgr_) {
        UPDATE_LOG_E("OTAStore::WriteECUSWData cfg_mgr_ is nullptr.");
        return false;
    }    

    std::string ecuSWStr = "";
    ecuSWStr.assign(ecuSWData.begin(), ecuSWData.end());
    cfg_mgr_->SetParam<std::string>("dids/F188", ecuSWStr, ConfigPersistType::CONFIG_SYNC_PERSIST);

    return true;
}

bool
OTAStore::ReadECUVersionData(std::vector<uint8_t>& ecuVersionData)
{
    UPDATE_LOG_I("OTAStore::ReadECUVersionData");

    if (nullptr == cfg_mgr_) {
        UPDATE_LOG_E("OTAStore::ReadECUVersionData cfg_mgr_ is nullptr.");
        return false;
    }

    std::vector<uint8_t> data;
    ReadDataFromCfg("dids/F1C0", UpdateManagerInfoDataType::kASCII, data);

    if (0 == data.size()) {
        ReadDataFromCfg("dids/F188", UpdateManagerInfoDataType::kASCII, data);
        if (0 == data.size()) {
            UPDATE_LOG_W("OTAStore::ReadECUVersionData get data is empty.");
            return false;
        }
    }
    if (data.size() != ECU_VERSION_DATA_LENGTH) {
        UPDATE_LOG_E("OTAStore::ReadECUVersionData ecu_version_data_ size error. correct f1c0_data_.size: %u , cfg get data.size: %zu "
                    ,ECU_VERSION_DATA_LENGTH, data.size());
        return false;
    }

    for (auto& item : data) {
        ecuVersionData.push_back(item);
    }
    return true;
}

bool
OTAStore::WriteECUVersionData(const std::vector<uint8_t>& ecuVersionData)
{
    UPDATE_LOG_I("OTAStore::WriteECUVersionData");
    if (ecuVersionData.size() != ECU_VERSION_DATA_LENGTH) {
        UPDATE_LOG_E("OTAStore::WriteECUVersionData ecuVersionData size error. need ecuVersionData.size: %u, ecuVersionData.size: %zu."
                    , ECU_VERSION_DATA_LENGTH, ecuVersionData.size());
        return false;
    }

    if (nullptr == cfg_mgr_) {
        UPDATE_LOG_E("OTAStore::WriteECUVersionData cfg_mgr_ is nullptr.");
        return false;
    }

    std::string ecuVersionStr = "";
    ecuVersionStr.assign(ecuVersionData.begin(), ecuVersionData.end());
    cfg_mgr_->SetParam<std::string>("dids/F1C0", ecuVersionStr, ConfigPersistType::CONFIG_SYNC_PERSIST);

    return true;
}

bool 
OTAStore::WriteTesterSNData(const std::vector<uint8_t>& testerSNData)
{
    UPDATE_LOG_I("OTAStore::WriteTesterSNData");
    if (testerSNData.size() != TESTER_SN_DATA_LENGTH) {
        UPDATE_LOG_E("OTAStore::WriteTesterSNData cfgWordData size error. need testerSNData.size: %u, cfgWordData.size: %zu."
                    , TESTER_SN_DATA_LENGTH, testerSNData.size());
        return false;
    }

    if (nullptr == cfg_mgr_) {
        UPDATE_LOG_E("OTAStore::WriteTesterSNData cfg_mgr_ is nullptr.");
        return false;
    }

    if (!(DataCheck(CheckType::kNumberAndLetterAndSymbol, testerSNData))) {
        UPDATE_LOG_E("DiagServerStoredInfo::WriteTesterSNData testerSNData data check failed.");
        return false;
    }
    std::string testerSNStr = "";
    testerSNStr.assign(testerSNData.begin(), testerSNData.end());
    cfg_mgr_->SetParam<std::string>("dids/F198", testerSNStr, ConfigPersistType::CONFIG_SYNC_PERSIST);
    return true;
}

bool
OTAStore::WriteProgrammingDateData(const std::vector<uint8_t>& programmingDateData)
{
    UPDATE_LOG_I("OTAStore::WriteProgrammingDateData");
    if (programmingDateData.size() != PROGRAMMING_DATE_DATA_LENGTH) {
        UPDATE_LOG_E("OTAStore::WriteProgrammingDateData programmingDateData size error. need programmingDateData.size: %u, programmingDateData.size: %zu."
                    , PROGRAMMING_DATE_DATA_LENGTH, programmingDateData.size());
        return false;
    }

    if (nullptr == cfg_mgr_) {
        UPDATE_LOG_E("OTAStore::WriteProgrammingDateData cfg_mgr_ is nullptr.");
        return false;
    }

    UM_DEBUG << "write dids/F199 str is : " << UM_UINT8_VEC_TO_HEX_STRING(programmingDateData);
    cfg_mgr_->SetParam<std::string>("dids/F199", UM_UINT8_VEC_TO_HEX_STRING(programmingDateData), ConfigPersistType::CONFIG_SYNC_PERSIST);
    return true;
}

bool 
OTAStore::ReadSocVersionData(std::string& socVersion)
{
    UPDATE_LOG_I("OTAStore::ReadSocVersionData");

    if (nullptr == cfg_mgr_) {
        UPDATE_LOG_E("OTAStore::ReadSocVersionData cfg_mgr_ is nullptr.");
        return false;
    }

    std::string data{};
    cfg_mgr_->GetParam<std::string>("version/SOC", data);

    if (0 == data.size()) {
        UPDATE_LOG_W("OTAStore::ReadSocVersionData get data is empty.");
        return false;
    }

    socVersion = data;
    return true;
}

bool 
OTAStore::WriteSocVersionData(const std::string& socVersion)
{
    UPDATE_LOG_I("OTAStore::WriteSocVersionData");

    if (nullptr == cfg_mgr_) {
        UPDATE_LOG_E("OTAStore::WriteSocVersionData cfg_mgr_ is nullptr.");
        return false;
    }

    // TODO 校验格式

    cfg_mgr_->SetParam<std::string>("version/SOC", socVersion, ConfigPersistType::CONFIG_SYNC_PERSIST);
    return true;
}

bool 
OTAStore::ReadDsvVersionData(std::string& dsvVersion)
{
    UPDATE_LOG_I("OTAStore::ReadDsvVersionData");

    if (nullptr == cfg_mgr_) {
        UPDATE_LOG_E("OTAStore::ReadDsvVersionData cfg_mgr_ is nullptr.");
        return false;
    }

    std::string data{};
    cfg_mgr_->GetParam<std::string>("version/DSV", data);

    if (0 == data.size()) {
        UPDATE_LOG_W("OTAStore::ReadDsvVersionData get data is empty.");
        return false;
    }

    dsvVersion = data;
    return true;
}

bool 
OTAStore::WriteDsvVersionData(const std::string& dsvVersion)
{
    UPDATE_LOG_I("OTAStore::WriteDsvVersionData");

    if (nullptr == cfg_mgr_) {
        UPDATE_LOG_E("OTAStore::WriteDsvVersionData cfg_mgr_ is nullptr.");
        return false;
    }

    // TODO 校验格式

    cfg_mgr_->SetParam<std::string>("version/DSV", dsvVersion, ConfigPersistType::CONFIG_SYNC_PERSIST);
    return true;
}

bool 
OTAStore::ReadMcuVersionData(std::string& mcuVersion)
{
    UPDATE_LOG_I("OTAStore::ReadMcuVersionData");

    if (nullptr == cfg_mgr_) {
        UPDATE_LOG_E("OTAStore::ReadMcuVersionData cfg_mgr_ is nullptr.");
        return false;
    }

    std::string data{};
    cfg_mgr_->GetParam<std::string>("version/MCU", data);

    if (0 == data.size()) {
        UPDATE_LOG_W("OTAStore::ReadMcuVersionData get data is empty.");
        return false;
    }

    mcuVersion = data;
    return true;
}

bool
OTAStore::WriteMcuVersionData(const std::string& mcuVersion)
{
    UPDATE_LOG_I("OTAStore::WriteMcuVersionData");

    if (nullptr == cfg_mgr_) {
        UPDATE_LOG_E("OTAStore::WriteMcuVersionData cfg_mgr_ is nullptr.");
        return false;
    }

    // TODO 校验格式

    cfg_mgr_->SetParam<std::string>("version/MCU", mcuVersion, ConfigPersistType::CONFIG_SYNC_PERSIST);
    return true;
}

bool 
OTAStore::ReadSwitchVersionData(std::string& switchVersion)
{
    UPDATE_LOG_I("OTAStore::ReadSwitchVersionData");

    if (nullptr == cfg_mgr_) {
        UPDATE_LOG_E("OTAStore::ReadSwitchVersionData cfg_mgr_ is nullptr.");
        return false;
    }

    std::string data{};
    cfg_mgr_->GetParam<std::string>("version/SWT", data);

    if (0 == data.size()) {
        UPDATE_LOG_W("OTAStore::ReadSwitchVersionData get data is empty.");
        return false;
    }

    switchVersion = data;
    return true;
}

bool
OTAStore::WriteSwitchVersionData(const std::string& switchVersion)
{
    UPDATE_LOG_I("OTAStore::WriteSwitchVersionData");

    if (nullptr == cfg_mgr_) {
        UPDATE_LOG_E("OTAStore::WriteSwitchVersionData cfg_mgr_ is nullptr.");
        return false;
    }

    // TODO 校验格式

    cfg_mgr_->SetParam<std::string>("version/SWT", switchVersion, ConfigPersistType::CONFIG_SYNC_PERSIST);
    return true;
}

bool 
OTAStore::WriteSensorVersionData(const std::string& sensorName, const std::string& sensorVersion)
{
    UPDATE_LOG_I("OTAStore::WriteSensorVersionData");

    if (nullptr == cfg_mgr_) {
        UPDATE_LOG_E("OTAStore::WriteSensorVersionData cfg_mgr_ is nullptr.");
        return false;
    }

    // TODO 校验格式
    std::string sensorKey = "version/" + sensorName;
    UM_DEBUG << "cfg write key is : " << sensorKey;
    cfg_mgr_->SetParam<std::string>(sensorKey, sensorVersion, ConfigPersistType::CONFIG_SYNC_PERSIST);
    return true;
}

bool 
OTAStore::ReadSensorVersionData(const std::string& sensorName, std::string& sensorVersion)
{
    UPDATE_LOG_I("OTAStore::ReadSensorVersionData");

    if (nullptr == cfg_mgr_) {
        UPDATE_LOG_E("OTAStore::ReadSensorVersionData cfg_mgr_ is nullptr.");
        return false;
    }
    std::string sensorKey = "version/" + sensorName;
    UM_DEBUG << "cfg read key is : " << sensorKey;
    std::string data{};
    cfg_mgr_->GetParam<std::string>(sensorKey, data);

    if (0 == data.size()) {
        UPDATE_LOG_W("OTAStore::ReadSensorVersionData get data is empty.");
        return false;
    }

    sensorVersion = std::move(data);
    return true;
}

bool 
OTAStore::WriteDynamicSensorVersionData(const std::string& sensorName, const std::string& sensorVersion)
{
    UPDATE_LOG_I("OTAStore::WriteSensorVersionData");

    if (nullptr == cfg_mgr_) {
        UPDATE_LOG_E("OTAStore::WriteSensorVersionData cfg_mgr_ is nullptr.");
        return false;
    }

    std::string sensorKey = "version/" + sensorName + "_DYNAMIC";
    UM_DEBUG << "cfg write key is : " << sensorKey;
    cfg_mgr_->SetParam<std::string>(sensorKey, sensorVersion, ConfigPersistType::CONFIG_NO_PERSIST);
    return true;
}

bool 
OTAStore::ReadPackageNameData(std::string& pkgName)
{
    UPDATE_LOG_I("OTAStore::ReadPackageNameData");

    if (nullptr == cfg_mgr_) {
        UPDATE_LOG_E("OTAStore::ReadPackageNameData cfg_mgr_ is nullptr.");
        return false;
    }

    std::string data{};
    cfg_mgr_->GetParam<std::string>("ota/PKGName", data);

    if (0 == data.size()) {
        UPDATE_LOG_W("OTAStore::ReadPackageNameData get data is empty.");
        return false;
    }

    pkgName = data;
    return true;
}

bool 
OTAStore::WritePackageNameData(const std::string& pkgName)
{
    UPDATE_LOG_I("OTAStore::WritePackageNameData");

    if (nullptr == cfg_mgr_) {
        UPDATE_LOG_E("OTAStore::WritePackageNameData cfg_mgr_ is nullptr.");
        return false;
    }

    cfg_mgr_->SetParam<std::string>("ota/PKGName", pkgName, ConfigPersistType::CONFIG_SYNC_PERSIST);
    return true;
}


bool 
OTAStore::ReadCmdFlagData(bool& cmdFlag)
{
    UPDATE_LOG_I("OTAStore::ReadCmdFlagData");

    if (nullptr == cfg_mgr_) {
        UPDATE_LOG_E("OTAStore::ReadCmdFlagData cfg_mgr_ is nullptr.");
        return false;
    }

    bool data{};
    cfg_mgr_->GetParam<bool>("ota/CmdFlag", data);

    cmdFlag = data;
    return true;
}
    
bool 
OTAStore::WriteCmdFlagData(const bool& cmdFlag)
{
    UPDATE_LOG_I("OTAStore::WriteCmdFlagData");

    if (nullptr == cfg_mgr_) {
        UPDATE_LOG_E("OTAStore::WriteCmdFlagData cfg_mgr_ is nullptr.");
        return false;
    }

    cfg_mgr_->SetParam<bool>("ota/CmdFlag", cmdFlag, ConfigPersistType::CONFIG_SYNC_PERSIST);
    return true;
}

bool
OTAStore::DataCheck(const CheckType& type, const std::vector<uint8_t>& data)
{
    for (auto& item : data) {
        switch (type)
        {
            case CheckType::kNumber:
                if (!(item >= '0' && item <= '9')) {
                    return false;
                }

                break;
            case CheckType::kLetter:
                if (!((item >= 'a' && item <= 'z') || (item >= 'A' && item <= 'Z'))) {
                    return false;
                }

                break;
            case CheckType::kNumberAndLetter:
                if (!((item >= '0' && item <= '9') || (item >= 'a' && item <= 'z') || (item >= 'A' && item <= 'Z'))) {
                    return false;
                }

                break;
            case CheckType::kNumberAndLetterAndSymbol:
                if (!((item >= '0' && item <= '9') || (item >= 'a' && item <= 'z') || (item >= 'A' && item <= 'Z') || item == '.' || item == ' ')) {
                    return false;
                }

                break;
            default:
                return false;
                break;
        }
    }

    return true;
}

void
OTAStore::ReadDataFromCfg(const std::string& dataKey, const UpdateManagerInfoDataType& dataType, std::vector<uint8_t>& data)
{
    if ("" == dataKey) {
        UM_WARN << "OTAStore::ReadDataFromCfg dataKey is empty.";
        return;
    }

    std::string dataStr = "";
    cfg_mgr_->GetParam<std::string>(dataKey, dataStr);

    if ("" == dataStr) {
        UM_WARN << "OTAStore::ReadDataFromCfg get dataStr is empty.";
        return;
    }

    if (dataType == UpdateManagerInfoDataType::kASCII) {
        data.assign(dataStr.begin(), dataStr.end());
    }
    else {
        auto vec = Split(dataStr);
        for (auto& item : vec) {
            data.push_back(static_cast<uint8_t>(std::strtoul(item.c_str(), 0, 16)));
        }
    }
}

std::vector<std::string>
OTAStore::Split(const std::string& inputStr, const std::string& regexStr)
{
    std::regex re(regexStr);
    std::sregex_token_iterator first {inputStr.begin(), inputStr.end(), re, -1}, last;
    return {first, last};
}

}  // namespace update
}  // namespace netaos
}  // namespace hozon
