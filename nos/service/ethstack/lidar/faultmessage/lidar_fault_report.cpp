#include "common/logger.h"
#include "phm/include/phm_client.h"
#include "phm/include/phm_def.h"
#include "lidar_fault_report.h"
#include <cstdint>
#include <string>
#include <unordered_map>


namespace hozon {
namespace ethstack {
namespace lidar {

LidarFaultReport::LidarFaultReport()    
: phm_client_ptr_(new netaos::phm::PHMClient())
{
}

LidarFaultReport& LidarFaultReport::Instance() {
    static LidarFaultReport instance;
    return instance;
}

void LidarFaultReport::Init() {
    if (nullptr != phm_client_ptr_) {
        phm_client_ptr_->Init();
    }
}

int32_t LidarFaultReport::ReportLidarSensorFault(enum LidarSensorFaultObject errObject,uint8_t faultStatus) {
    if (nullptr == phm_client_ptr_) {
        LIDAR_LOG_ERROR << "LidarFaultReport::ReportLidarSensorFault error: phm_client_ptr_ is nullptr!";
        return -1;
    }

    LIDAR_LOG_DEBUG << "ReportLidarSensorFault() case : " << (uint8_t)errObject << " Start Report... ";
    netaos::phm::SendFault_t lidarFaultInfo(LIDAR_SENSOR_FAULT_ID,(uint8_t)errObject,faultStatus);
    int32_t result = phm_client_ptr_->ReportFault(lidarFaultInfo);
    LIDAR_LOG_DEBUG << "ReportLidarSensorFault() case : " << (uint8_t)errObject << " Complete Report. ";
    return result;
}

int32_t LidarFaultReport::ReportAbstractFault(enum AbstractFaultObject errObject,uint8_t faultStatus) {
    if (nullptr == phm_client_ptr_) {
        LIDAR_LOG_ERROR << "LidarFaultReport::ReportAbstractFault error: phm_client_ptr_ is nullptr!";
        return -1;
    }

    LIDAR_LOG_DEBUG << "ReportAbstractFault() case : " << (uint8_t)errObject << " Start Report... ";
    netaos::phm::SendFault_t lidarFaultInfo(ABSTRACT_FAULT_ID,(uint8_t)errObject,faultStatus);
    int32_t result = phm_client_ptr_->ReportFault(lidarFaultInfo);
    LIDAR_LOG_DEBUG << "ReportAbstractFault() case : " << (uint8_t)errObject << " Complete Report. ";
    return result;
}

int32_t LidarFaultReport::ReportCommunicationFault(enum CommunicationFaultObject errObject,uint8_t faultStatus) {
    if (nullptr == phm_client_ptr_) {
        LIDAR_LOG_ERROR << "LidarFaultReport::ReportCommunicationFault error: phm_client_ptr_ is nullptr!";
        return -1;
    }

    LIDAR_LOG_DEBUG << "ReportCommunicationFault() case : " << (uint8_t)errObject << " Start Report... ";
    netaos::phm::SendFault_t lidarFaultInfo(COMMUNICATION_FAULT_ID,(uint8_t)errObject,faultStatus);
    int32_t result = phm_client_ptr_->ReportFault(lidarFaultInfo);
    LIDAR_LOG_DEBUG << "ReportCommunicationFault() case : " << (uint8_t)errObject << " Complete Report. ";
    return result;
}

void LidarFaultReport::DeInit() {
    if (nullptr != phm_client_ptr_) {
        phm_client_ptr_->Deinit();
        delete phm_client_ptr_;
        phm_client_ptr_ = nullptr;
    }
}


}
}
}