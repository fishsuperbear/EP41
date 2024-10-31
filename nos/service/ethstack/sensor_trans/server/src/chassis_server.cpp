
#include <sys/types.h>
#include "chassis_server.h"
#include "chassis_server_data.h"
#include "logger.h"

namespace hozon {
namespace netaos {
namespace sensor {
int32_t ChassisServer::Process(const std::shared_ptr<ChassisOtaMethod> req_data,
                    std::shared_ptr<ChassisOtaMethod> resp_data) {
    SENSOR_LOG_INFO << "chassis server receive request.";
    std::shared_ptr<ChassisOtaStruct> _chassis_ota_method = ChassisServerData::GetInstance().GetData();
    resp_data->stamp().sec(_chassis_ota_method->stamp.sec); 
    resp_data->stamp().nsec(_chassis_ota_method->stamp.nsec);
    resp_data->gnss_stamp().sec(_chassis_ota_method->gnss_stamp.sec);
    resp_data->gnss_stamp().nsec(_chassis_ota_method->gnss_stamp.nsec);
    resp_data->ESC_VehicleSpeedValid(_chassis_ota_method->ESC_VehicleSpeedValid);
    resp_data->ESC_VehicleSpeed(_chassis_ota_method->ESC_VehicleSpeed);
    resp_data->BDCS10_AC_OutsideTempValid(_chassis_ota_method->BDCS10_AC_OutsideTempValid);
    resp_data->BDCS10_AC_OutsideTemp(_chassis_ota_method->BDCS10_AC_OutsideTemp);
    resp_data->BDCS1_PowerManageMode(_chassis_ota_method->BDCS1_PowerManageMode);

    resp_data->ICU2_Odometer(_chassis_ota_method->ICU2_Odometer);
    resp_data->Ignition_status(_chassis_ota_method->Ignition_status);
    resp_data->VCU_ActGearPosition_Valid(_chassis_ota_method->VCU_ActGearPosition_Valid);
    resp_data->VCU_ActGearPosition(_chassis_ota_method->VCU_ActGearPosition);

    // // for test
    // resp_data->BDCS1_PowerManageMode(1);
    // SENSOR_LOG_INFO << "resp_data->BDCS1_PowerManageMode " << resp_data->BDCS1_PowerManageMode();
    return 0;
}

}
}
}




