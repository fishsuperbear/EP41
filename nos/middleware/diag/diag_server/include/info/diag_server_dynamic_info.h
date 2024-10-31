/*
* Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
* Description: diag server dynamic info
*/

#ifndef DIAG_SERVER_DYNAMIC_INFO_H
#define DIAG_SERVER_DYNAMIC_INFO_H

#include <mutex>
#include <vector>
#include <iostream>
#include "devm/include/devm_device_info.h"
#include "diag/diag_server/include/info/diag_server_chassis_info.h"

using namespace hozon::netaos::devm;

namespace hozon {
namespace netaos {
namespace diag {

class DiagServerDynamicInfo {

public:
    static DiagServerDynamicInfo* getInstance();

    void Init();
    void DeInit();

    // Update Install Status [0x0107]
    bool ReadInstallStatus(std::vector<uint8_t>& installStatusData);

    // Power Supply Voltage [0x0112]
    bool ReadPowerSupplyVoltage(std::vector<uint8_t>& powerVoltage);

    // Odometer [0xE101]
    bool ReadOdometerValue(std::vector<uint8_t>& odometerValue);

    // Vehicle Speed [0xB100]
    bool ReadVehicleSpeed(std::vector<uint8_t>& vehicleSpeed);

    // Ignition Status [D001]
    bool ReadIgnitionStatus(std::vector<uint8_t>& ignitionStatus);

    // Time [0xF020]
    bool ReadTime(std::vector<uint8_t>& timeData);

    // PKI Certificate Application Status [0x8001]
    bool ReadPKIApplyStatus(std::vector<uint8_t>& pkiApplyStatus);

    // ADASFront30CameraCalibrationStatus [0xF103"]
    bool ReadADASF30CameraCalibrationStatus(std::vector<uint8_t>& f30CalibrationStatus);

    // ADASFront120CameraCalibrationStatus [0xF104"]
    bool ReadADASF120CameraCalibrationStatus(std::vector<uint8_t>& f120CalibrationStatus);

    // ADASFLCameraCalibrationStatus [0xF105"]
    bool ReadADASFLCameraCalibrationStatus(std::vector<uint8_t>& fLCalibrationStatus);

    // ADASFRCameraCalibrationStatus [0xF106"]
    bool ReadADASFRCameraCalibrationStatus(std::vector<uint8_t>& fRCalibrationStatus);

    // ADASRLCameraCalibrationStatus [0xF107"]
    bool ReadADASRLCameraCalibrationStatus(std::vector<uint8_t>& rLCalibrationStatus);

    // ADASRRCameraCalibrationStatus [0xF108"]
    bool ReadADASRRCameraCalibrationStatus(std::vector<uint8_t>& rRCalibrationStatus);

    // ADASRearCameraCalibrationStatus [0xF109"]
    bool ReadADASRearCameraCalibrationStatus(std::vector<uint8_t>& rearCalibrationStatus);

    // ADASFront30AndFront120CameraCoordinatedCalibrationStatus [0xF117"]
    bool ReadADASF30AndF120CameraCoordinatedCalibrationStatus(std::vector<uint8_t>& f30AndF120CalibrationStatus);

    // ADASFront120AndRLCameraCoordinatedCalibrationStatus [0xF118"]
    bool ReadADASF120AndRLCameraCoordinatedCalibrationStatus(std::vector<uint8_t>& f120AndRLCalibrationStatus);

    // ADASFront120AndRRCameraCoordinatedCalibrationStatus [0xF119"]
    bool ReadADASF120AndRRCameraCoordinatedCalibrationStatus(std::vector<uint8_t>& f120AndRRCalibrationStatus);

    // ADASFLAndRLCameraCoordinatedCalibrationStatus [0xF120"]
    bool ReadADASFLAndRLCameraCoordinatedCalibrationStatus(std::vector<uint8_t>& fLAndRLCalibrationStatus);

    // ADASFRAndRRCameraCoordinatedCalibrationStatus [0xF121"]
    bool ReadADASFRAndRRCameraCoordinatedCalibrationStatus(std::vector<uint8_t>& fRAndRRCalibrationStatus);

    // ADASFLAndRearCameraCoordinatedCalibrationStatus [0xF122"]
    bool ReadADASFLAndRearCameraCoordinatedCalibrationStatus(std::vector<uint8_t>& fLAndRearCalibrationStatus);

    // ADASFRAndRearCameraCoordinatedCalibrationStatus [0xF123"]
    bool ReadADASFRAndRearCameraCoordinatedCalibrationStatus(std::vector<uint8_t>& fRAndRearCalibrationStatus);

    // AfterSalesADASFront30CameraCalibrationStatus [0xF110"]
    bool ReadAfterSalesADASF30CameraCalibrationStatus(std::vector<uint8_t>& afterSalesF30CalibrationStatus);

    // AfterSalesADASFront120CameraCalibrationStatus [0xF111"]
    bool ReadAfterSalesADASF120CameraCalibrationStatus(std::vector<uint8_t>& afterSalesF120CalibrationStatus);

    // AfterSalesADASFLCameraCalibrationStatus [0xF112"]
    bool ReadAfterSalesADASFLCameraCalibrationStatus(std::vector<uint8_t>& afterSalesFLCalibrationStatus);

    // AfterSalesADASFRCameraCalibrationStatus [0xF113"]
    bool ReadAfterSalesADASFRCameraCalibrationStatus(std::vector<uint8_t>& afterSalesFRCalibrationStatus);

    // AfterSalesADASRLCameraCalibrationStatus [0xF114"]
    bool ReadAfterSalesADASRLCameraCalibrationStatus(std::vector<uint8_t>& afterSalesRLCalibrationStatus);

    // AfterSalesADASRRCameraCalibrationStatus [0xF115"]
    bool ReadAfterSalesADASRRCameraCalibrationStatus(std::vector<uint8_t>& afterSalesRRCalibrationStatus);

    // AfterSalesADASRearCameraCalibrationStatus [0xF116"]
    bool ReadAfterSalesADASRearCameraCalibrationStatus(std::vector<uint8_t>& afterSalesRearCalibrationStatus);

    bool ReadMcuDidInfo(uint16_t did, std::vector<uint8_t>& mcuDataInfo);

private:
    DiagServerDynamicInfo();
    DiagServerDynamicInfo(const DiagServerDynamicInfo &);
    DiagServerDynamicInfo & operator = (const DiagServerDynamicInfo &);

private:
    static DiagServerDynamicInfo* instance_;
    static std::mutex mtx_;

    DiagServerChassisInfo* chassis_info_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_DYNAMIC_INFO_H
