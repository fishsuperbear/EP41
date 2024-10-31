/*
* Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
* Description: diag server chassis info
*/

#ifndef DIAG_SERVER_CHASSIS_INFO_H
#define DIAG_SERVER_CHASSIS_INFO_H

#include <mutex>
#include "diag/diag_server/include/common/diag_server_def.h"

namespace hozon {
namespace netaos {
namespace diag {

class DiagServerChassisInfo {

public:
    static DiagServerChassisInfo* getInstance();

    void Init();
    void DeInit();

    void UpdateChassisInfo();
    DiagServerChassisData GetChassisData() {return chassis_data_;}

    // gear display
    void SetGearDisplay(const uint8_t gearDisplay) {chassis_data_.gearDisplay = gearDisplay;}
    uint8_t GetGearDisplay() {return chassis_data_.gearDisplay;}

    // outside temp
    void SetOutsideTemp(const float outsideTemp) {chassis_data_.outsideTemp = outsideTemp;}
    float GetOutsideTemp() {return chassis_data_.outsideTemp;}

    // odo meter
    void SetOdometer(const float odometer) {chassis_data_.odometer = odometer;}
    float GetOdometer() {return chassis_data_.odometer;}

    // power mode
    void SetPowerMode(const uint8_t powerMode) {chassis_data_.powerMode = powerMode;}
    uint8_t GetPowerMode() {return chassis_data_.powerMode;}

    // ig status
    void SetIgStatus(const uint8_t igStatus) {chassis_data_.igStatus = igStatus;}
    uint8_t GetIgStatus() {return chassis_data_.igStatus;}

    // vehicle speed valid
    void SetVehicleSpeedValid(const bool vehicleSpeedValid) {chassis_data_.vehicleSpeedValid = vehicleSpeedValid;}
    bool GetVehicleSpeedValid() {return chassis_data_.vehicleSpeedValid;}

    // vehicle speed
    void SetVehicleSpeed(const double vehicleSpeed) {chassis_data_.vehicleSpeed = vehicleSpeed;}
    double GetVehicleSpeed() {return chassis_data_.vehicleSpeed;}

private:
    DiagServerChassisInfo();
    DiagServerChassisInfo(const DiagServerChassisInfo &);
    DiagServerChassisInfo & operator = (const DiagServerChassisInfo &);

private:
    static DiagServerChassisInfo* instance_;
    static std::mutex mtx_;

    DiagServerChassisData chassis_data_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_CHASSIS_INFO_H