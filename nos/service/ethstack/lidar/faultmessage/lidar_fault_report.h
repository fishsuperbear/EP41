#pragma once 

#include <cstdint>
#include "common/logger.h"
#include "phm/include/phm_client.h"
#include "phm/include/phm_def.h"



namespace hozon {
namespace ethstack {
namespace lidar {

const uint32_t LIDAR_SENSOR_FAULT_ID = 4700;
const uint32_t ABSTRACT_FAULT_ID = 4701;
const uint32_t COMMUNICATION_FAULT_ID = 4702;
const uint8_t CURRENT_FAULT = 1;


enum class LidarSensorFaultObject {
    PRE_PERFORMANCE_DEGRADATION_ERROR = 1,
    PERFORMANCE_DEGRADATION_ERROR,
    PRE_SHUTDOWN_ERROR,
    SHUTDOWN_OR_OUTPUT_UNTRUSTED_ERROR,
    PRE_RESET_ERROR,
    RESET_ERROR,
    MILD_BLOCK_ERROR,
    FULL_BLOCK_ERROR,
    HARDWARE_ERROR,
};

enum class AbstractFaultObject {
    CONFIG_LOAD_ERROR = 1,
    SOCKET_ERROR,
    CM_INIT_ERROR,
    DATA_HANDLE_ACQUISITION_FAILED,
};

enum class CommunicationFaultObject {
    ERROR_FRAME_OCCUR = 1,
    CHECKSUM_FAILED,
    COMM_LOST,
    SEQ_EXEPTION,
    DATA_FIELD_ERROR,
};


class LidarFaultReport {
public:
    static LidarFaultReport& Instance();
    ~LidarFaultReport(){};

    void Init();
    void DeInit();

    int32_t ReportLidarSensorFault(enum LidarSensorFaultObject errObject,uint8_t faultStatus);
    int32_t ReportAbstractFault(enum AbstractFaultObject errObject,uint8_t faultStatus);
    int32_t ReportCommunicationFault(enum CommunicationFaultObject errObject,uint8_t faultStatus);


private:
    void PhmServiceAvailableCallback(const bool bResult);
    // void FaultReceiveCallback(const netaos::phm::ReceiveFault_t& fault);

private:
    LidarFaultReport();
    netaos::phm::PHMClient* phm_client_ptr_;
     bool phmServerAvailable_ = false;
    
};






}
}
}