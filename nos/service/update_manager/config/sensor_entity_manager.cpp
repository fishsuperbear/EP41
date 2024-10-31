#include "update_manager/config/sensor_entity_manager.h"
#include "update_manager/log/update_manager_logger.h"

namespace hozon {
namespace netaos {
namespace update {

SensorEntityManager* SensorEntityManager::m_pInstance = nullptr;
std::mutex SensorEntityManager::m_mtx;

SensorEntityManager::SensorEntityManager()
{
}

SensorEntityManager::~SensorEntityManager()
{
}

SensorEntityManager*
SensorEntityManager::Instance()
{
    if (nullptr == m_pInstance) {
        std::unique_lock<std::mutex> lck(m_mtx);
        if (nullptr == m_pInstance) {
            m_pInstance = new SensorEntityManager();
        }
    }

    return m_pInstance;
}

void
SensorEntityManager::Init()
{
    UM_INFO << "SensorEntityManager::Init.";
    UM_INFO << "SensorEntityManager::Init Done.";
}

void
SensorEntityManager::Deinit()
{
    UM_INFO << "SensorEntityManager::Deinit.";
    if (m_pInstance != nullptr) {
        delete m_pInstance;
        m_pInstance = nullptr;
    }
    UM_INFO << "SensorEntityManager::Deinit Done.";
}

bool 
SensorEntityManager::ParseEntityByPartNum(const std::string& sensorName, const std::string& partNum, const std::vector<SensorEntityInfo_t> entitys)
{
    UM_INFO << "SensorEntityManager::ParseEntityByPartNum. sensorName is : " << sensorName << " ,partNum is : " << partNum;
    for (const auto& entity : entitys) {
        size_t found = partNum.find(entity.partNumber);
        if (found != std::string::npos) {
            UM_DEBUG << "SensorEntityManager: find entity succ, sensor name is : " << sensorName;
            sensor_info_map_[sensorName] = entity;
        }
    }
    return true;
}

std::string 
SensorEntityManager::GetPartNumber(const std::string& sensorName)
{
    UM_INFO << "SensorEntityManager::GetPartNumber.";
    auto it = sensor_info_map_.find(sensorName);

    if (it != sensor_info_map_.end()) {
        UM_DEBUG << "SensorEntityManager: GetPartNumber is : " << it->second.partNumber;
        return it->second.partNumber;
    } else {
        UM_ERROR << "Sensor with name '" << sensorName << "' not found.";
        return "";
    }
}

std::string 
SensorEntityManager::GetSupplierCode(const std::string& sensorName)
{
    UM_INFO << "SensorEntityManager::GetSupplierCode.";
    auto it = sensor_info_map_.find(sensorName);

    if (it != sensor_info_map_.end()) {
        UM_DEBUG << "SensorEntityManager: GetSupplierCode is : " << it->second.supplierCode;
        return it->second.supplierCode;
    } else {
        UM_ERROR << "Sensor with name '" << sensorName << "' not found.";
        return "";
    }
}

std::string 
SensorEntityManager::GetTargetVersion(const std::string& sensorName)
{
    UM_INFO << "SensorEntityManager::GetTargetVersion.";
    auto it = sensor_info_map_.find(sensorName);

    if (it != sensor_info_map_.end()) {
        UM_DEBUG << "SensorEntityManager: GetTargetVersion is : " << it->second.targetVersion;
        return it->second.targetVersion;
    } else {
        UM_ERROR << "Sensor with name '" << sensorName << "' not found.";
        return "";
    }
}

std::string 
SensorEntityManager::GetFlashDriverFirmwareName(const std::string& sensorName)
{
    UM_INFO << "SensorEntityManager::GetFlashDriverFirmwareName.";
    auto it = sensor_info_map_.find(sensorName);

    if (it != sensor_info_map_.end()) {
        UM_DEBUG << "SensorEntityManager: GetFlashDriverFirmwareName is : " << it->second.flashDriverFirmwareName;
        return it->second.flashDriverFirmwareName;
    } else {
        UM_ERROR << "Sensor with name '" << sensorName << "' not found.";
        return "";
    }
}

std::string 
SensorEntityManager::GetCalFirmwareName(const std::string& sensorName)
{
    UM_INFO << "SensorEntityManager::GetCalFirmwareName.";
    auto it = sensor_info_map_.find(sensorName);

    if (it != sensor_info_map_.end()) {
        UM_DEBUG << "SensorEntityManager: GetCalFirmwareName is : " << it->second.calFirmwareName;
        return it->second.calFirmwareName;
    } else {
        UM_ERROR << "Sensor with name '" << sensorName << "' not found.";
        return "";
    }
}

std::string 
SensorEntityManager::GetProcessFileName(const std::string& sensorName)
{
    // 如果查不到，返回第一个元素的信息
    UM_INFO << "SensorEntityManager::GetProcessFileName.";
    auto it = sensor_info_map_.find(sensorName);

    if (it != sensor_info_map_.end()) {
        UM_DEBUG << "SensorEntityManager: GetProcessFileName is : " << it->second.processFileName;
        return it->second.processFileName;
    } else {
        UM_ERROR << "Sensor with name '" << sensorName << "' not found.";
        return "";
    }
}

bool 
SensorEntityManager::ResetSensorEntityInfo()
{
    UM_INFO << "SensorEntityManager::ResetSensorEntityInfo.";
    sensor_info_map_.clear();
    return true;
}

}  // namespace update
}  // namespace netaos
}  // namespace hozon
