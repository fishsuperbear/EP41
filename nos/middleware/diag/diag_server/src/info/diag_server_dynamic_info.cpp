/*
* Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
* Description: diag server dynamic info
*/

#include "diag/diag_server/include/publish/diag_server_uds_pub.h"
#include "diag/diag_server/include/info/diag_server_dynamic_info.h"
#include "diag/diag_server/include/common/diag_server_logger.h"
#include "diag/diag_server/include/info/cfg_data.h"
#include "cfg/include/config_param.h"
#include <sys/time.h>
#include <time.h>

namespace hozon {
namespace netaos {
namespace diag {

DiagServerDynamicInfo* DiagServerDynamicInfo::instance_ = nullptr;
std::mutex DiagServerDynamicInfo::mtx_;

DiagServerDynamicInfo::DiagServerDynamicInfo()
: chassis_info_(DiagServerChassisInfo::getInstance())
{

}

DiagServerDynamicInfo*
DiagServerDynamicInfo::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new DiagServerDynamicInfo();
        }
    }

    return instance_;
}

void
DiagServerDynamicInfo::Init()
{
    DG_INFO << "DiagServerDynamicInfo::Init";
}

void
DiagServerDynamicInfo::DeInit()
{
    DG_INFO << "DiagServerDynamicInfo::DeInit";
    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
}

// Update Install Status [0x0107]
bool
DiagServerDynamicInfo::ReadInstallStatus(std::vector<uint8_t>& installStatusData)
{
    DG_DEBUG << "DiagServerStoredInfo::ReadInstallStatus";
    // TO DO: get install status from updatemansger
    installStatusData.push_back(0x00);
    return true;
}

// Power Supply Voltage [0x0112]
bool
DiagServerDynamicInfo::ReadPowerSupplyVoltage(std::vector<uint8_t>& powerVoltage)
{
    DG_INFO << "DiagServerDynamicInfo::ReadPowerSupplyVoltage";
    std::shared_ptr<DevmClientDeviceInfo> devm_ptr = std::make_shared<DevmClientDeviceInfo>();
    VoltageData data;
    if (!(devm_ptr->GetVoltage(data))) {
        DG_ERROR << "DiagServerStoredInfo::ReadPowerSupplyVoltage get PowerSupplyVoltage fail!";
        return false;
    }

    if (!(data.kl15 > 0)) {
        DG_ERROR << "DiagServerStoredInfo::ReadPowerSupplyVoltage get PowerSupplyVoltage value error!";
        return false;
    }

    DG_INFO << "DiagServerDynamicInfo::ReadPowerSupplyVoltage data: " << data.kl30;
    powerVoltage.push_back(static_cast<uint8_t>(data.kl30 * 10));
    return true;
}

// Odometer [0xE101]
bool
DiagServerDynamicInfo::ReadOdometerValue(std::vector<uint8_t>& odometerValue)
{
    DG_INFO << "DiagServerDynamicInfo::ReadOdometerValue";
    if (nullptr == chassis_info_) {
        DG_ERROR << "DiagServerStoredInfo::ReadOdometerValue chassis_info_ is nullptr.";
        return false;
    }

    chassis_info_->UpdateChassisInfo();
    float tempValue = chassis_info_->GetOdometer();
    if (tempValue < 0) {
        DG_WARN << "DiagServerStoredInfo::ReadOdometerValue get data is invalid, data: " << tempValue;
        return false;
    }

    uint32_t data = tempValue * 10;
    odometerValue.push_back(static_cast<uint8_t>(data >> 16));
    odometerValue.push_back(static_cast<uint8_t>(data >> 8));
    odometerValue.push_back(static_cast<uint8_t>(data));

    return true;
}

// Vehicle Speed [0xB100]
bool
DiagServerDynamicInfo::ReadVehicleSpeed(std::vector<uint8_t>& vehicleSpeed)
{
    DG_INFO << "DiagServerDynamicInfo::ReadVehicleSpeed";
    if (nullptr == chassis_info_) {
        DG_ERROR << "DiagServerStoredInfo::ReadVehicleSpeed chassis_info_ is nullptr.";
        return false;
    }

    chassis_info_->UpdateChassisInfo();
    double tempValue = chassis_info_->GetVehicleSpeed();
    if (tempValue < 0) {
        DG_WARN << "DiagServerStoredInfo::ReadVehicleSpeed get data is invalid, data: " << tempValue;
        return false;
    }

    uint16_t data = tempValue * 100;
    vehicleSpeed.push_back(static_cast<uint8_t>(data >> 8));
    vehicleSpeed.push_back(static_cast<uint8_t>(data));

    return true;
}

// Ignition Status [D001]
bool
DiagServerDynamicInfo::ReadIgnitionStatus(std::vector<uint8_t>& ignitionStatus)
{
    DG_INFO << "DiagServerDynamicInfo::ReadIgnitionStatus";
    if (nullptr == chassis_info_) {
        DG_ERROR << "DiagServerStoredInfo::ReadIgnitionStatus chassis_info_ is nullptr.";
        return false;
    }

    chassis_info_->UpdateChassisInfo();
    ignitionStatus.push_back(chassis_info_->GetIgStatus());

    return true;
}

// Time [0xF020]
bool
DiagServerDynamicInfo::ReadTime(std::vector<uint8_t>& timeData)
{
    DG_INFO << "DiagServerDynamicInfo::ReadTime";
    uint16_t year = 2018;
    uint8_t month = 1;
    uint8_t day = 1;
    uint8_t hour = 0;
    uint8_t minute = 0;
    uint8_t second = 0;

    do  {
        struct timespec ts;
        if (clock_gettime(CLOCK_REALTIME,&ts)) {
            break;
        }

        struct tm* timeinfo = localtime(&ts.tv_sec);
        if (nullptr ==  timeinfo) {
            break;
        }
        year = timeinfo->tm_year + 1900;
        month = timeinfo->tm_mon + 1;
        day = timeinfo->tm_mday;
        hour = timeinfo->tm_hour;
        minute = timeinfo->tm_min;
        second = timeinfo->tm_sec;
    } while(0);

    timeData.push_back(static_cast<uint8_t>(year >> 8));
    timeData.push_back(static_cast<uint8_t>(year));
    timeData.push_back(month);
    timeData.push_back(day);
    timeData.push_back(hour);
    timeData.push_back(minute);
    timeData.push_back(second);

    return true;
}

// PKI Certificate Application Status [0x8001]
bool
DiagServerDynamicInfo::ReadPKIApplyStatus(std::vector<uint8_t>& pkiApplyStatus)
{
    DG_INFO << "DiagServerDynamicInfo::ReadTime";
    ConfigParam* cfg_mgr = ConfigParam::Instance();
    if (nullptr == cfg_mgr) {
        DG_ERROR << "DiagServerDynamicInfo::ReadPKIApplyStatus cfg_mgr is nullptr.";
        return false;
    }

    uint8_t data = 0xFF;
    std::string dataStr = "";
    cfg_mgr->Init(1000);
    CfgResultCode res = cfg_mgr->GetParam<uint8_t>("pki/status", data);
    if (res != CONFIG_OK) {
        dataStr = CfgValueInfo::getInstance()->GetCfgValueFromFile("/cfg/pki/pki.json", "pki/status");
        if ("" == dataStr) {
            DG_ERROR << "DiagServerDynamicInfo::ReadPKIApplyStatus read pki failed";
            return false;
        }
        data = static_cast<uint8_t>(std::strtoul(dataStr.c_str(), 0, 16));
    }

    if (0xFF != data) {
        pkiApplyStatus.push_back(data);
        return true;
    }

    return false;
}

// ADASFront30CameraCalibrationStatus [0xF103"]
bool
DiagServerDynamicInfo::ReadADASF30CameraCalibrationStatus(std::vector<uint8_t>& f30CalibrationStatus)
{
    DG_INFO << "DiagServerDynamicInfo::ReadTime";
    //TO DO
    for (uint16_t i = 0; i < 14; ++i) {
        f30CalibrationStatus.push_back(0x00);
    }

    return true;
}

// ADASFront120CameraCalibrationStatus [0xF104"]
bool
DiagServerDynamicInfo::ReadADASF120CameraCalibrationStatus(std::vector<uint8_t>& f120CalibrationStatus)
{
    DG_INFO << "DiagServerDynamicInfo::ReadTime";
    //TO DO
    for (uint16_t i = 0; i < 14; ++i) {
        f120CalibrationStatus.push_back(0x00);
    }

    return true;
}

// ADASFLCameraCalibrationStatus [0xF105"]
bool
DiagServerDynamicInfo::ReadADASFLCameraCalibrationStatus(std::vector<uint8_t>& fLCalibrationStatus)
{
    DG_INFO << "DiagServerDynamicInfo::ReadTime";
    //TO DO
    for (uint16_t i = 0; i < 14; ++i) {
        fLCalibrationStatus.push_back(0x00);
    }

    return true;
}

// ADASFRCameraCalibrationStatus [0xF106"]
bool
DiagServerDynamicInfo::ReadADASFRCameraCalibrationStatus(std::vector<uint8_t>& fRCalibrationStatus)
{
    DG_INFO << "DiagServerDynamicInfo::ReadTime";
    //TO DO
    for (uint16_t i = 0; i < 14; ++i) {
        fRCalibrationStatus.push_back(0x00);
    }

    return true;
}

// ADASRLCameraCalibrationStatus [0xF107"]
bool
DiagServerDynamicInfo::ReadADASRLCameraCalibrationStatus(std::vector<uint8_t>& rLCalibrationStatus)
{
    DG_INFO << "DiagServerDynamicInfo::ReadTime";
    //TO DO
    for (uint16_t i = 0; i < 14; ++i) {
        rLCalibrationStatus.push_back(0x00);
    }

    return true;
}

// ADASRRCameraCalibrationStatus [0xF108"]
bool
DiagServerDynamicInfo::ReadADASRRCameraCalibrationStatus(std::vector<uint8_t>& rRCalibrationStatus)
{
    DG_INFO << "DiagServerDynamicInfo::ReadTime";
    //TO DO
    for (uint16_t i = 0; i < 14; ++i) {
        rRCalibrationStatus.push_back(0x00);
    }

    return true;
}

// ADASRearCameraCalibrationStatus [0xF109"]
bool
DiagServerDynamicInfo::ReadADASRearCameraCalibrationStatus(std::vector<uint8_t>& rearCalibrationStatus)
{
    DG_INFO << "DiagServerDynamicInfo::ReadTime";
    //TO DO
    for (uint16_t i = 0; i < 14; ++i) {
        rearCalibrationStatus.push_back(0x00);
    }

    return true;
}

// ADASFront30AndFront120CameraCoordinatedCalibrationStatus [0xF117"]
bool
DiagServerDynamicInfo::ReadADASF30AndF120CameraCoordinatedCalibrationStatus(std::vector<uint8_t>& f30AndF120CalibrationStatus)
{
    DG_INFO << "DiagServerDynamicInfo::ReadTime";
    //TO DO
    for (uint16_t i = 0; i < 14; ++i) {
        f30AndF120CalibrationStatus.push_back(0x00);
    }

    return true;
}

// ADASFront120AndRLCameraCoordinatedCalibrationStatus [0xF118"]
bool
DiagServerDynamicInfo::ReadADASF120AndRLCameraCoordinatedCalibrationStatus(std::vector<uint8_t>& f120AndRLCalibrationStatus)
{
    DG_INFO << "DiagServerDynamicInfo::ReadTime";
    //TO DO
    for (uint16_t i = 0; i < 14; ++i) {
        f120AndRLCalibrationStatus.push_back(0x00);
    }

    return true;
}

// ADASFront120AndRRCameraCoordinatedCalibrationStatus [0xF119"]
bool
DiagServerDynamicInfo::ReadADASF120AndRRCameraCoordinatedCalibrationStatus(std::vector<uint8_t>& f120AndRRCalibrationStatus)
{
    DG_INFO << "DiagServerDynamicInfo::ReadTime";
    //TO DO
    for (uint16_t i = 0; i < 14; ++i) {
        f120AndRRCalibrationStatus.push_back(0x00);
    }

    return true;
}

// ADASFLAndRLCameraCoordinatedCalibrationStatus [0xF120"]
bool
DiagServerDynamicInfo::ReadADASFLAndRLCameraCoordinatedCalibrationStatus(std::vector<uint8_t>& fLAndRLCalibrationStatus)
{
    DG_INFO << "DiagServerDynamicInfo::ReadTime";
    //TO DO
    for (uint16_t i = 0; i < 14; ++i) {
        fLAndRLCalibrationStatus.push_back(0x00);
    }

    return true;
}

// ADASFRAndRRCameraCoordinatedCalibrationStatus [0xF121"]
bool
DiagServerDynamicInfo::ReadADASFRAndRRCameraCoordinatedCalibrationStatus(std::vector<uint8_t>& fRAndRRCalibrationStatus)
{
    DG_INFO << "DiagServerDynamicInfo::ReadTime";
    //TO DO
    for (uint16_t i = 0; i < 14; ++i) {
        fRAndRRCalibrationStatus.push_back(0x00);
    }

    return true;
}

// ADASFLAndRearCameraCoordinatedCalibrationStatus [0xF122"]
bool
DiagServerDynamicInfo::ReadADASFLAndRearCameraCoordinatedCalibrationStatus(std::vector<uint8_t>& fLAndRearCalibrationStatus)
{
    DG_INFO << "DiagServerDynamicInfo::ReadTime";
    //TO DO
    for (uint16_t i = 0; i < 14; ++i) {
        fLAndRearCalibrationStatus.push_back(0x00);
    }

    return true;
}

// ADASFRAndRearCameraCoordinatedCalibrationStatus [0xF123"]
bool
DiagServerDynamicInfo::ReadADASFRAndRearCameraCoordinatedCalibrationStatus(std::vector<uint8_t>& fRAndRearCalibrationStatus)
{
    DG_INFO << "DiagServerDynamicInfo::ReadTime";
    //TO DO
    for (uint16_t i = 0; i < 14; ++i) {
        fRAndRearCalibrationStatus.push_back(0x00);
    }

    return true;
}

// AfterSalesADASFront30CameraCalibrationStatus [0xF110"]
bool
DiagServerDynamicInfo::ReadAfterSalesADASF30CameraCalibrationStatus(std::vector<uint8_t>& afterSalesF30CalibrationStatus)
{
    DG_INFO << "DiagServerDynamicInfo::ReadTime";
    //TO DO
    for (uint16_t i = 0; i < 14; ++i) {
        afterSalesF30CalibrationStatus.push_back(0x00);
    }

    return true;
}

// AfterSalesADASFront120CameraCalibrationStatus [0xF111"]
bool
DiagServerDynamicInfo::ReadAfterSalesADASF120CameraCalibrationStatus(std::vector<uint8_t>& afterSalesF120CalibrationStatus)
{
    DG_INFO << "DiagServerDynamicInfo::ReadTime";
    //TO DO
    for (uint16_t i = 0; i < 14; ++i) {
        afterSalesF120CalibrationStatus.push_back(0x00);
    }

    return true;
}

// AfterSalesADASFLCameraCalibrationStatus [0xF112"]
bool
DiagServerDynamicInfo::ReadAfterSalesADASFLCameraCalibrationStatus(std::vector<uint8_t>& afterSalesFLCalibrationStatus)
{
    DG_INFO << "DiagServerDynamicInfo::ReadTime";
    //TO DO
    for (uint16_t i = 0; i < 14; ++i) {
        afterSalesFLCalibrationStatus.push_back(0x00);
    }

    return true;
}

// AfterSalesADASFRCameraCalibrationStatus [0xF113"]
bool
DiagServerDynamicInfo::ReadAfterSalesADASFRCameraCalibrationStatus(std::vector<uint8_t>& afterSalesFRCalibrationStatus)
{
    DG_INFO << "DiagServerDynamicInfo::ReadTime";
    //TO DO
    for (uint16_t i = 0; i < 14; ++i) {
        afterSalesFRCalibrationStatus.push_back(0x00);
    }

    return true;
}

// AfterSalesADASRLCameraCalibrationStatus [0xF114"]
bool
DiagServerDynamicInfo::ReadAfterSalesADASRLCameraCalibrationStatus(std::vector<uint8_t>& afterSalesRLCalibrationStatus)
{
    DG_INFO << "DiagServerDynamicInfo::ReadTime";
    //TO DO
    for (uint16_t i = 0; i < 14; ++i) {
        afterSalesRLCalibrationStatus.push_back(0x00);
    }

    return true;
}

// AfterSalesADASRRCameraCalibrationStatus [0xF115"]
bool
DiagServerDynamicInfo::ReadAfterSalesADASRRCameraCalibrationStatus(std::vector<uint8_t>& afterSalesRRCalibrationStatus)
{
    DG_INFO << "DiagServerDynamicInfo::ReadTime";
    //TO DO
    for (uint16_t i = 0; i < 14; ++i) {
        afterSalesRRCalibrationStatus.push_back(0x00);
    }

    return true;
}

// AfterSalesADASRearCameraCalibrationStatus [0xF116"]
bool
DiagServerDynamicInfo::ReadAfterSalesADASRearCameraCalibrationStatus(std::vector<uint8_t>& afterSalesRearCalibrationStatus)
{
    DG_INFO << "DiagServerDynamicInfo::ReadTime";
    //TO DO
    for (uint16_t i = 0; i < 14; ++i) {
        afterSalesRearCalibrationStatus.push_back(0x00);
    }

    return true;
}

// ReadMcuDisInfo [0xF1C1 ~ 0xF1C6]
bool
DiagServerDynamicInfo::ReadMcuDidInfo(uint16_t did, std::vector<uint8_t>& mcuDataInfo)
{
    DG_INFO << "DiagServerDynamicInfo::ReadMcuDidInfo";
    bool ret = false;
#ifdef BUILD_FOR_ORIN
    ret = DiagServerUdsPub::getInstance()->GetMcuDidsInfo(did, mcuDataInfo);
#endif
    return ret;
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
