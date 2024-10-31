/*
 * Copyright (c) Hozon SOC Co., Ltd. 2023-2023. All rights reserved.
 *
 * Description: doip client socket
 */

#ifndef DEVM_DATA_GATHERING_H_
#define DEVM_DATA_GATHERING_H_
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "cm/include/method.h"
#include "devm_server_logger.h"
#include "cfg/include/config_param.h"

namespace hozon {
namespace netaos {
namespace devm_server {

using namespace hozon::netaos::cfg;
enum DiagServerInfoDataType {
    kNumber = 0x00,
    kLetter = 0x01,
    kNumberAndLetter = 0x02,
    kNumberAndLetterAndSymbol = 0x03,
    kHEX = 0x04,
    kBCD = 0x05,
    kASCII = 0x06
};
enum DiagServerSessionCode {
    kDefaultSession = 0x01,
    kProgrammingSession = 0x02,
    kExtendedSession = 0x03
};

const std::string HARDWARE_NUM = "H1.10";
const std::string BOOT_SW = "00.01.00";
const std::string ECU_PART_NUM = "C40-3686999  ";
const std::vector<uint8_t> ECU_TYPE = {0x4C, 0x49, 0x4E, 0x55, 0x58, 0xFF, 0xFF, 0xFF};

const uint8_t VIN_DATA_LENGTH = 17;
const uint8_t VEHICLE_CFG_WORD_DATA_LENGTH = 58;
const uint8_t ECU_SW_DATA_LENGTH = 8;
const uint8_t TESTER_SN_DATA_LENGTH = 10;
const uint8_t PROGRAMMING_DATE_DATA_LENGTH = 4;
const uint8_t ECU_TYPE_DATA_LENGTH = 8;
const uint8_t ECU_INSTALL_DATE_LENGTH = 4;
const uint8_t ESK_NUMBER_LENGTH = 16;
const uint8_t BOOT_SW_DATA_LENGTH = 8;
const uint8_t ECU_SOFTWARE_NUMBER_LENGTH = 8;
const uint8_t HARDWARE_NUM_LENGTH = 5;
const uint8_t ECU_PART_NUM_LENGTH = 13;
const uint8_t SYS_SUPPLIER_ID_LENGTH = 3;
const uint8_t ECU_MANUFACTURE_DATE_LENGTH = 4;
const uint8_t ECU_SERIAL_NUMBER_LENGTH = 18;
const uint8_t FACTORY_MODE_LENGTH = 1;
const uint8_t SOC_VER_LENGTH = 41;
const uint8_t MCU_VER_LENGTH = 40;
const uint8_t XPC_VER_LENGTH = 26;

struct DiagServerChassisData {
    uint8_t gearDisplay;
    float outsideTemp;
    float odometer;
    uint8_t powerMode;
    uint8_t igStatus;
    bool vehicleSpeedValid;
    double vehicleSpeed;
};

#include <mutex>
class DiagServerChassisInfo {

public:
    static DiagServerChassisInfo* getInstance();

    void Init() {}
    static void DeInit()
    {
        DEVM_LOG_INFO << "DiagServerChassisInfo::DeInit";
        if (nullptr != instance_) {
            delete instance_;
            instance_ = nullptr;
        }
    }


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
    DiagServerChassisInfo() { memset(&chassis_data_, 0x00, sizeof(chassis_data_)); }
    DiagServerChassisInfo(const DiagServerChassisInfo &);
    DiagServerChassisInfo & operator= (const DiagServerChassisInfo &);


    static DiagServerChassisInfo* instance_;
    static std::mutex mtx_;

    DiagServerChassisData chassis_data_;
};









/* devm_server data gather class */
class DevmDataGathering {
public:
    static DevmDataGathering& GetInstance() {
        static DevmDataGathering instance;
        return instance;
    }
    ~DevmDataGathering(){};

    void Init();
    void DeInit();
    bool ReadEcuSoftwareNumber2(std::vector<uint8_t>& ecuSoftwareNumber);
    std::string ReadCfgWord();

    bool GetValueWithDid(uint16_t did, std::vector<uint8_t>& data);

    // Vehicle Configuration Word [0xF170]
    bool ReadVehicleCfgWordData(std::vector<uint8_t>& cfgWordData);
    bool WriteVehicleCfgWordData(const std::vector<uint8_t>& cfgWordData);

    // Vehicle Identification Number [0xF190]
    bool ReadVinData(std::vector<uint8_t>& vinData);
    bool WriteVinData(const std::vector<uint8_t>& vinData);

    // Vehicle ECU Software Number Data Identifier [0xF188]
    bool ReadECUSWData(std::vector<uint8_t>& ecuSWData);

    // Tester Serial Number Data Identifier [0xF198]
    bool ReadTesterSNData(std::vector<uint8_t>& testerSNData);
    bool WriteTesterSNData(const std::vector<uint8_t>& testerSNData);

    // Programming Date Data Identifier [0xF199]
    bool ReadProgrammingDateData(std::vector<uint8_t>& programmingDateData);
    bool WriteProgrammingDateData(const std::vector<uint8_t>& programmingDateData);

    // Ecu Type [0x0110]
    bool ReadEcuType(std::vector<uint8_t>& ecuTypeData);

    // ECU Install Date [0xF19D]
    bool ReadInstallDate(std::vector<uint8_t>& installDate);
    bool WriteInstallDate(const std::vector<uint8_t>& installDate);

    // Boot SW Identifier [0xF180]
    bool ReadBootSWId(std::vector<uint8_t>& bootSWId);

    // Active Diagnostic Session [0xF186]
    bool ReadCurrDiagSession(std::vector<uint8_t>& currDiagSession);

    // Vehicle Manufacturer Spare Part Number [0xF187]
    bool ReadVehicleManufacturerSparePartNumber(std::vector<uint8_t>& vehiclePartNumber);

    // ECU Software Number [0xF1B0]
    bool ReadEcuSoftwareNumber(std::vector<uint8_t>& ecuSoftwareNumber);

    // System Supplier Identifier [0xF18A]
    bool ReadSystemSupplierId(std::vector<uint8_t>& sysSupplierId);

    // ECU Manufacture Date [0xF18B]
    bool ReadEcuManufactureDate(std::vector<uint8_t>& ecuManufactureDate);

    // ECU Serial Number [0xF18C]
    bool ReadEcuSerialNumber(std::vector<uint8_t>& ecuSerialNumber);

    // ECU Hardware Version [0xF191]
    bool ReadEcuHardwareVersion(std::vector<uint8_t>& ecuHardwareVersion);

    // ECU Hardware Number [0xF1BF]
    bool ReadEcuHardwareNumber(std::vector<uint8_t>& ecuHardwareNumber);

    // ECU Software Assembly Part Number [0xF1D0]
    bool ReadEcuSoftwareAssemblyPartNumber(std::vector<uint8_t>& ecuPartNumber);

    // ESK Number [0x900F]
    bool ReadEskNumber(std::vector<uint8_t>& eskNumber);
    bool WriteEskNumber(const std::vector<uint8_t>& eskNumber);

    // Domain controller all sensor versions [0xF1E0]
    bool ReadAllSensorVersions(std::vector<uint8_t>& allSensorVersions);

    // Orin Version [0xF1E1]
    bool ReadOrinVersion(std::vector<uint8_t>& orinVersion);

    // SOC Version [0xF1E2]
    bool ReadSOCVersion(std::vector<uint8_t>& socVersion);

    // MCU Version [0xF1E3]
    bool ReadMCUVersion(std::vector<uint8_t>& mcuVersion);

    // Factory Mode [0x2910]
    bool ReadFactoryMode(std::vector<uint8_t>& mode);
    bool WriteFactoryMode(const std::vector<uint8_t>& mode);

    // sid 85 save cfg
    void saveControlDtcStatusToCFG(const uint8_t status);

    // calibrate status save cfg
    void saveCalibrateStatusToCFG(const std::vector<uint8_t>& data);

    static bool DataCheck(const DiagServerInfoDataType& type, const std::vector<uint8_t>& data);

    void ReadDataFromCfg(const std::string& dataKey, const DiagServerInfoDataType& dataType, std::vector<uint8_t>& data);

    static std::vector<std::string> Split(const std::string& inputStr, const std::string& regexStr = " ");



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
private:
    DevmDataGathering()
    : cfg_mgr_(ConfigParam::Instance())
    , vin_number_("0123456789abcdefg")
    , cfg_word_("123")
    , chassis_info_(DiagServerChassisInfo::getInstance()) {}

    ConfigParam* cfg_mgr_;
    std::string vin_number_;
    std::string cfg_word_;

    std::vector<uint8_t> tester_sn_data_;
    std::vector<uint8_t> programming_date_data_;
    std::vector<uint8_t> ecu_type_data_;
    std::vector<uint8_t> install_date_data_;
    std::vector<uint8_t> boot_sw_data_;
    std::vector<uint8_t> hardware_num_data_;
    std::vector<uint8_t> ecu_part_num_data_;

    DiagServerChassisInfo *chassis_info_;
};




}  // namespace devm_server
}  // namespace netaos
}  // namespace hozon
#endif  // end of DEVM_DATA_GATHERING_H_
