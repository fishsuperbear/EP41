#include "fault_message_parse.h"
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>

#include "common/logger.h"
#include "lidar_fault_report.h"


namespace hozon {

namespace ethstack {

namespace lidar {



FaultMessageParser& FaultMessageParser::Instance() {
    static FaultMessageParser instance;
    return instance;
}

FaultMessageParser::FaultMessageParser()
    : CurrentStatus_(0)
{
}

FaultMessageParser::~FaultMessageParser(){

}

void FaultMessageParser::Parse(uint8_t* dataptr, uint32_t size){

    FaultMessage* frame = reinterpret_cast<FaultMessage*>(dataptr);
    if((frame->header.leader1 == 0xCD) && (frame->header.leader2 == 0xDC)){
        std::tm tm_time;
        tm_time.tm_year = frame->data_time[0];
        tm_time.tm_mon = frame->data_time[1] - 1;
        tm_time.tm_mday = frame->data_time[2];
        tm_time.tm_hour = frame->data_time[3];
        tm_time.tm_min = frame->data_time[4];
        tm_time.tm_sec = frame->data_time[5];
        tm_time.tm_isdst = 0;
        std::time_t timestamp = timegm(&tm_time);
        double time = timestamp + frame->timestamp * 1.0 / 1000000;
        LIDAR_LOG_TRACE << "Fault message time is : "<<std::to_string(time);
        uint8_t faultStatus = 0;

        //判断故障是current还是history
        if(frame->fault_code_type == 1){                //current
            faultStatus = frame->fault_code_type;
        }
        else if (frame->fault_code_type == 2) {         //history
            faultStatus = 0;
        }

        if((frame->fault_indication == 1) && (CurrentStatus_ != (uint8_t)LidarSensorFaultObject::MILD_BLOCK_ERROR)){
            LidarFaultReport::Instance().ReportLidarSensorFault(LidarSensorFaultObject::MILD_BLOCK_ERROR,faultStatus);
            CurrentStatus_ = (uint8_t)LidarSensorFaultObject::MILD_BLOCK_ERROR;
        }
        else if ((frame->lidar_fault_state == 2) && (CurrentStatus_ != (uint8_t)LidarSensorFaultObject::FULL_BLOCK_ERROR)) {
            LidarFaultReport::Instance().ReportLidarSensorFault(LidarSensorFaultObject::FULL_BLOCK_ERROR,faultStatus);
            CurrentStatus_ = (uint8_t)LidarSensorFaultObject::FULL_BLOCK_ERROR;
        }

        switch(frame->lidar_fault_state)
        {
        case 2:
            if(CurrentStatus_ != (uint8_t)LidarSensorFaultObject::PRE_PERFORMANCE_DEGRADATION_ERROR){
                LidarFaultReport::Instance().ReportLidarSensorFault(LidarSensorFaultObject::PRE_PERFORMANCE_DEGRADATION_ERROR,faultStatus);
                CurrentStatus_ = (uint8_t)LidarSensorFaultObject::PRE_PERFORMANCE_DEGRADATION_ERROR;
            }
            break;
        case 3:
            if(CurrentStatus_ != (uint8_t)LidarSensorFaultObject::PERFORMANCE_DEGRADATION_ERROR){
                LidarFaultReport::Instance().ReportLidarSensorFault(LidarSensorFaultObject::PERFORMANCE_DEGRADATION_ERROR,faultStatus);
                CurrentStatus_ = (uint8_t)LidarSensorFaultObject::PERFORMANCE_DEGRADATION_ERROR;
            }
            break;
        case 4:
            if(CurrentStatus_ != (uint8_t)LidarSensorFaultObject::PRE_SHUTDOWN_ERROR){
                LidarFaultReport::Instance().ReportLidarSensorFault(LidarSensorFaultObject::PRE_SHUTDOWN_ERROR,faultStatus);
                CurrentStatus_ = (uint8_t)LidarSensorFaultObject::PRE_SHUTDOWN_ERROR;
            }
            break;
        case 5:
            if(CurrentStatus_ != (uint8_t)LidarSensorFaultObject::SHUTDOWN_OR_OUTPUT_UNTRUSTED_ERROR){
                LidarFaultReport::Instance().ReportLidarSensorFault(LidarSensorFaultObject::SHUTDOWN_OR_OUTPUT_UNTRUSTED_ERROR,faultStatus);
                CurrentStatus_ = (uint8_t)LidarSensorFaultObject::SHUTDOWN_OR_OUTPUT_UNTRUSTED_ERROR;
            }
            break;
        case 6:
            if(CurrentStatus_ != (uint8_t)LidarSensorFaultObject::PRE_RESET_ERROR){
                LidarFaultReport::Instance().ReportLidarSensorFault(LidarSensorFaultObject::PRE_RESET_ERROR,faultStatus);
                CurrentStatus_ = (uint8_t)LidarSensorFaultObject::PRE_RESET_ERROR;
            }
            break;
        case 7:
            if(CurrentStatus_ != (uint8_t)LidarSensorFaultObject::RESET_ERROR){
                LidarFaultReport::Instance().ReportLidarSensorFault(LidarSensorFaultObject::RESET_ERROR,faultStatus);
                CurrentStatus_ = (uint8_t)LidarSensorFaultObject::RESET_ERROR;
            }
            break;
        default:
            break;
        }
                       
    }
}


}
}
}
