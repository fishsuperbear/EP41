#pragma once

#include <mutex>
#include "update_manager/common/data_def.h"
#include "update_manager/config/config_manager.h"

namespace hozon {
namespace netaos {
namespace update {

class SensorEntityManager {
public:
    
    static SensorEntityManager* Instance();

    void Init();
    void Deinit();
    bool ParseEntityByPartNum(const std::string& sensorName, const std::string& partNum, const std::vector<SensorEntityInfo_t> entitys);

    std::string GetPartNumber(const std::string& sensorName);
    std::string GetSupplierCode(const std::string& sensorName);
    std::string GetTargetVersion(const std::string& sensorName);
    std::string GetFlashDriverFirmwareName(const std::string& sensorName);
    std::string GetCalFirmwareName(const std::string& sensorName);
    std::string GetProcessFileName(const std::string& sensorName);

    bool ResetSensorEntityInfo();

private:
    SensorEntityManager();
    ~SensorEntityManager();
    SensorEntityManager(const SensorEntityManager &);
    SensorEntityManager & operator = (const SensorEntityManager &);

    static std::mutex m_mtx;
    static SensorEntityManager* m_pInstance;

    std::map<std::string, SensorEntityInfo_t> sensor_info_map_;
};

}  // namespace update
}  // namespace netaos
}  // namespace hozon
