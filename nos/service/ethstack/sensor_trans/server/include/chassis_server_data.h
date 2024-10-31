#pragma once
#include <cstdint>
#include <memory>
#include <mutex>


namespace hozon {
namespace netaos {
namespace  sensor {

struct Time { 
    /* data */
    uint64_t sec;
    uint64_t nsec;
};

struct ChassisOtaStruct {
    Time stamp;
    Time gnss_stamp;
    bool VCU_ActGearPosition_Valid;
    uint8_t VCU_ActGearPosition;
    bool ESC_VehicleSpeedValid;
    float ESC_VehicleSpeed;
    bool BDCS10_AC_OutsideTempValid;
    float BDCS10_AC_OutsideTemp;
    float ICU2_Odometer;
    uint8_t BDCS1_PowerManageMode;
    uint8_t Ignition_status;
};

class ChassisServerData {
public:
    static ChassisServerData& GetInstance() {
        static ChassisServerData instance;
        return instance;
    }
    ChassisServerData() = default;
    ~ChassisServerData() = default;

    int Write(std::shared_ptr<ChassisOtaStruct> data) {
        std::lock_guard<std::recursive_mutex> lck(_struct_mtx);
        _chassis_ota_data = data;
        return 0;
    }
    std::shared_ptr<ChassisOtaStruct> GetData() {
        std::lock_guard<std::recursive_mutex> lck(_struct_mtx);
        return _chassis_ota_data;
    }

private:
    std::shared_ptr<ChassisOtaStruct>  _chassis_ota_data = std::make_shared<ChassisOtaStruct>();
    std::recursive_mutex _struct_mtx;
};

}
}
}