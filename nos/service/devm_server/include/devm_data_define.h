/*
 * Copyright (c) Hozon SOC Co., Ltd. 2023-2023. All rights reserved.
 *
 * Description: doip client socket
 */


#pragma once

#include <iostream>


namespace hozon {
namespace netaos {
namespace devm_server {


// typedef enum DEVM_DATA_TYPE
// {
//     DEVM_DATA_VIN_NUM       = 0xf190,
//     DEVM_DATA_CFG_WORD       = 0xf170,
//     DEVM_DATA_VIN_NUM2       = 0x8002,
//     DEVM_DATA_VIN_NUM3       = 0x8003,
//     DEVM_DATA_VIN_NUM4       = 0x8004,
//     DEVM_DATA_VIN_NUM5       = 0x8005,
//     DEVM_DATA_VIN_NUM6       = 0x8006,
//     DEVM_DATA_VIN_NUM7       = 0x8007
// } devm_data_type_t;

typedef enum DEVM_DATA_TYPE
{
    DEVM_DATA_VEHICLE_CFG_WORD                  = 0xF170,
    DEVM_DATA_ECUSW_DATA                        = 0xF188,
    DEVM_DATA_VIN_DATA                          = 0xF190,
    DEVM_DATA_TESTER_SN                         = 0xF198,
    DEVM_DATA_PROGRAM_MINGDATE_DATA             = 0xF199,
    DEVM_DATA_INSTALL_STATUS                    = 0x0107,
    DEVM_DATA_ECU_TYPE                          = 0x0110,
    DEVM_DATA_BOOT_SW_ID                        = 0xF180,
    DEVM_DATA_CURR_DIAG_SESSION                 = 0xF186,
    DEVM_DATA_MANUFACTURER_NUMBER               = 0xF187,
    DEVM_DATA_ECU_SOFTWARE_NUMBER               = 0xF1B0,
    DEVM_DATA_SYSTEM_SUPPLIER_ID                = 0xF18A,
    DEVM_DATA_ECU_MANUFACTURE_DATE              = 0xF18B,
    DEVM_DATA_ECU_SERIAL_NUMBER                 = 0xF18C,
    DEVM_DATA_ECU_HARDWARE_VERSION              = 0xF191,
    DEVM_DATA_INSTALL_DATE                      = 0xF19D,
    DEVM_DATA_ECU_HARDWARE_NUMBER               = 0xF1BF,
    DEVM_DATA_ECU_SOFTWARE_NUMBER2              = 0xF1C0,
    DEVM_DATA_ECU_SOFTWARE_ASSEMBLY_NUMBER      = 0xF1D0,
    DEVM_DATA_ESK_NUMBER                        = 0x900F,
    DEVM_DATA_POWER_SUPPLY_VOLTAGE              = 0x0112,
    DEVM_DATA_ODOMETER_VALUE                    = 0xE101,
    DEVM_DATA_VEHICLE_SPEED                     = 0xB100,
    DEVM_DATA_IGNITION_STATUS                   = 0xD001,
    DEVM_DATA_READ_TIME                         = 0xF020,


    PKI_APPLY_STATUS                               = 0x8001,
    ADAS_F30_CALIBRATION_STATUS                    = 0xF103,
    ADAS_F120_CALIBRATION_STATUS                   = 0xF104,
    ADAS_FL_CALIBRATION_STATUS                     = 0xF105,
    ADAS_FR_CALIBRATION_STATUS                     = 0xF106,
    ADAS_RL_CALIBRATION_STATUS                     = 0xF107,
    ADAS_RR_CALIBRATION_STATUS                     = 0xF108,
    ADAS_REAR_CALIBRATION_STATUS                   = 0xF109,
    ADAS_F30ANDF120_COORDINATED_CALIBRATION_STATUS = 0xF117,
    ADAS_F120ANDRL_COORDINATED_CALIBRATION_STATUS  = 0xF118,
    ADAS_F120ANDRR_COORDINATED_CALIBRATION_STATUS  = 0xF119,
    ADAS_FLANDRL_COORDINATED_CALIBRATION_STATUS    = 0xF120,
    ADAS_FRANDRR_COORDINATED_CALIBRATION_STATUS    = 0xF121,
    ADAS_FLANDREAR_COORDINATED_CALIBRATION_STATUS  = 0xF122,
    ADAS_FRANDREAR_COORDINATED_CALIBRATION_STATUS  = 0xF123,
    AFTER_SALES_ADAS_F30_CALIBRATION_STATUS        = 0xF110,
    AFTER_SALES_ADAS_F120_CALIBRATION_STATUS       = 0xF111,
    AFTER_SALES_ADAS_FL_CALIBRATION_STATUS         = 0xF112,
    AFTER_SALES_ADAS_FR_CALIBRATION_STATUS         = 0xF113,
    AFTER_SALES_ADAS_RL_CALIBRATION_STATUS         = 0xF114,
    AFTER_SALES_ADAS_RR_CALIBRATION_STATUS         = 0xF115,
    AFTER_SALES_ADAS_REAR_CALIBRATION_STATUS       = 0xF116,
} devm_data_type_t;



// 定义电压信息
struct Voltage {
    float voltage_value; // 电压值，单位：伏特（V）
};

// 定义温度信息，多个核
struct Temperature {
    float temperature_core[100]; // 温度值，单位：摄氏度（°C）
};

// 定义版本信息
struct Version {
    std::string major_version; // 主版本号
    std::string minor_version; // 次版本号
    std::string patch_version; // 补丁版本号
};

// 定义产线信息
struct ProductionInfo {
    std::string vin; // VIN号
    std::string sn; // SN号
    std::string electronic_label; // 电子标签
    std::string production_date; // 生产日期
    std::string manufacturer_name; // 厂家名称
    std::string bom_number; // BOM编号
};

// 定义系统信息
struct SystemInfo {
    int cpu_cores; // CPU核心数
    int total_memory; // 总内存大小，单位：MB
    int disk_capacity; // 磁盘容量，单位：GB
    int emmc_lifespan; // eMMC寿命，单位：写入次数
};

// 定义升级信息
struct UpgradeInfo {
    struct Version app_version; // App版本
    struct Version middleware_version; // 中间件版本
    struct Version mcu_version; // MCU版本
    struct Version firmware_version; // 底层软件版本
};

// 定义电源管理信息
struct PowerManagementInfo {
    // 电源管理信息的具体字段，参考OTA获取的内容
};




#define SOC_VERSION_DYNAMIC     "version/SOC_DYNAMIC"
#define MCU_VERSION_DYNAMIC     "version/MCU_DYNAMIC"
#define DSV_VERSION_DYNAMIC     "version/DSV_DYNAMIC"
#define SWT_VERSION_DIDS        "version/SWT"
#define SWT_VERSION_DYNAMIC     "version/SWT_DYNAMIC"
#define USS_VERSION_DYNAMIC     "version/USS_DYNAMIC"
#define MAJOR_VERSION_DIDS      "dids/F1B0"
#define MAJOR_VERSION_DIDS_F1C0 "dids/F1C0"
#define MAJOR_VERSION_DIDS_F188 "dids/F188"
#define PART_NUMBER_DIDS        "dids/F187"
#define SYS_SUPPLIER_ID_DIDS    "dids/F18A"
#define ECU_MANUFACT_DATA_DIDS  "dids/F18B"
#define ECU_SERIAL_NUMBER_DIDS  "dids/F18C"
#define ECU_HARD_NUMBER_DIDS    "dids/F191"



// class CfgDataImpl {
// public:
//     static CfgDataImpl* getInstance() {
//         static CfgDataImpl instance;
//         return &instance;
//     }
//     ConfigParam* cfg_mgr_{};
// private:
//     CfgDataImpl() {
//         cfg_mgr_ = ConfigParam::Instance();
//     }
//     ~CfgDataImpl(){}
// };







// 定义整体设备信息结构体
// struct DeviceInfo {
//     struct Voltage voltage_info; // 电压信息
//     struct Temperature temperature_info; // 温度信息
//     struct ProductionInfo production_info; // 产线信息
//     struct SystemInfo system_info; // 系统信息
//     struct UpgradeInfo upgrade_info; // 升级信息
//     struct PowerManagementInfo power_management_info; // 电源管理信息
// };


}  // namespace devm_server
}  // namespace netaos
}  // namespace hozon

