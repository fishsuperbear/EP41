/*
* Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
* Description: diag server stored info
*/

#ifndef DIAG_SERVER_STORED_INFO_H
#define DIAG_SERVER_STORED_INFO_H

#include <mutex>
#include <iostream>
#include <vector>
#include <algorithm>

#include "cfg/include/config_param.h"
#include "diag/diag_server/include/common/diag_server_def.h"

namespace hozon {
namespace netaos {
namespace diag {

using namespace hozon::netaos::cfg;

class DiagServerStoredInfo {

public:
    static DiagServerStoredInfo* getInstance();

    void Init();
    void DeInit();

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

    // keys file transfer completed save cfg
    void KeysFileTransferCompletedToCFG();

    static bool DataCheck(const DiagServerInfoDataType& type, const std::vector<uint8_t>& data);

private:
    void ReadDataFromCfg(const std::string& dataKey, const DiagServerInfoDataType& dataType, std::vector<uint8_t>& data);

    static std::vector<std::string> Split(const std::string& inputStr, const std::string& regexStr = " ");

private:
    DiagServerStoredInfo();
    DiagServerStoredInfo(const DiagServerStoredInfo &);
    DiagServerStoredInfo & operator = (const DiagServerStoredInfo &);

private:
    static DiagServerStoredInfo* instance_;
    static std::mutex mtx_;

    std::vector<uint8_t> tester_sn_data_;
    std::vector<uint8_t> programming_date_data_;
    std::vector<uint8_t> ecu_type_data_;
    std::vector<uint8_t> install_date_data_;
    std::vector<uint8_t> boot_sw_data_;
    std::vector<uint8_t> hardware_num_data_;
    std::vector<uint8_t> ecu_part_num_data_;

    ConfigParam* cfg_mgr_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_STORED_INFO_H
