#pragma once

#include <string>
#include <vector>
#include <mutex>
#include "cfg/include/config_param.h"
#include "devm_data_define.h"
#include "devm_server_logger.h"
#include "function_statistics.h"

namespace hozon {
namespace netaos {
namespace devm_server {

struct DeviceInfo {
    std::string major_version;
    std::string soc_version;
    std::string mcu_version;
    std::string swt_version;
    std::string dsv_version;
    std::string uss_version;
    std::string soc_type = "OrinX";
    std::string mcu_type = "TC397";
    std::string switch_type = "Marvell";
    std::map<std::string, std::string> sensor_version;
};

struct Devicestatus {
    std::string soc_status;
    std::string mcu_status;
    std::map<std::string, std::string> camera_status;
    std::map<std::string, std::string> lidar_status;
    std::map<std::string, std::string> radar_status;
    std::map<std::string, std::string> imu_status;
    std::map<std::string, std::string> uss_status;
};

struct TemperatureData {
    float soc_temp;     //soc核温
    float mcu_temp;     //mcu核温
    float ext0_temp;    //tmp451近soc温度
    float ext1_temp;    //tmp451近mcu温度
};
struct VoltageData {
    int32_t kl15_vol;     //KL15电压
    float kl30_vol;      //KL30电压
};


const std::map<std::string, std::string> version_tables = {
    {"version/LIDAR_DYNAMIC",       "LIDAR"            },
    {"version/SRR_FL_DYNAMIC",      "SRR_FL"           },
    {"version/SRR_FR_DYNAMIC",      "SRR_FR"           },
    {"version/SRR_RL_DYNAMIC",      "SRR_RL"           },
    {"version/SRR_RR_DYNAMIC",      "SRR_RR"           },
    {"version/CAM_FRONT_30",        "CAM_FRONT_30"     },
    {"version/CAM_FRONT_120",       "CAM_FRONT_120"    },
    {"version/CAM_REAR",            "CAM_REAR"         },
    {"version/CAM_LF",              "CAM_LF"           },
    {"version/CAM_LR",              "CAM_LR"           },
    {"version/CAM_RF",              "CAM_RF"           },
    {"version/CAM_RR",              "CAM_RR"           },
    {"version/CAM_FRONT_AVM",       "CAM_FRONT_AVM"    },
    {"version/CAM_LEFT_AVM",        "CAM_LEFT_AVM"     },
    {"version/CAM_REAR_AVM",        "CAM_REAR_AVM"     },
    {"version/CAM_RIGHT_AVM",       "CAM_RIGHT_AVM"    },
};

const std::map<std::string, std::string> camera_status_tables = {
    {"system/cam_front_30_status",  "cam_front_30"     },
    {"system/cam_front_120_status", "cam_front_120"    },
    {"system/cam_rear_status",      "cam_rear"         },
    {"system/cam_lf_status",        "cam_lf"           },
    {"system/cam_lr_status",        "cam_lr"           },
    {"system/cam_rf_status",        "cam_rf"           },
    {"system/cam_rr_status",        "cam_rr"           },
    {"system/cam_front_avm_status", "cam_front_avm"    },
    {"system/cam_left_avm_status",  "cam_left_avm"     },
    {"system/cam_rear_avm_status",  "cam_rear_avm"     },
    {"system/cam_right_avm_status", "cam_right_avm"    }
};
const std::map<std::string, std::string> lidar_status_tables = {
    {"system/lidar_status",         "lidar"            }
};
const std::map<std::string, std::string> radar_status_tables = {
    {"system/front_radar_status",   "front_radar"      },
    {"system/fl_radar_status",      "fl_radar"         },
    {"system/fr_radar_status",      "fr_radar"         },
    {"system/rl_radar_status",      "rl_radar"         },
    {"system/rr_radar_status",      "rr_radar"         }
};
const std::map<std::string, std::string> uss_status_tables = {
    {"system/uss_fls_status",       "uss_fls"          },
    {"system/uss_frs_status",       "uss_frs"          },
    {"system/uss_rls_status",       "uss_rls"          },
    {"system/uss_rrs_status",       "uss_rrs"          },
    {"system/uss_flc_status",       "uss_flc"          },
    {"system/uss_flm_status",       "uss_flm"          },
    {"system/uss_frm_status",       "uss_frm"          },
    {"system/uss_frc_status",       "uss_frc"          },
    {"system/uss_rlc_status",       "uss_rlc"          },
    {"system/uss_rlm_status",       "uss_rlm"          },
    {"system/uss_rrm_status",       "uss_rrm"          },
    {"system/uss_rrc_status",       "uss_rrc"          }
};


using namespace hozon::netaos::cfg;
// device info/status 全局变量类，只做set_data，get_data
class DeviceInfomation {
public:
    static DeviceInfomation* getInstance() {
        static DeviceInfomation instance;
        return &instance;
    }
    void Init() {
        FunctionStatistics func("DeviceInfomation::Init, ");
        ConfigParam::Instance()->GetParam(SOC_VERSION_DYNAMIC, data_.soc_version);
        ConfigParam::Instance()->GetParam(MCU_VERSION_DYNAMIC, data_.mcu_version);
        ConfigParam::Instance()->GetParam(DSV_VERSION_DYNAMIC, data_.dsv_version);
        ConfigParam::Instance()->GetParam(SWT_VERSION_DYNAMIC, data_.swt_version);
        ConfigParam::Instance()->GetParam(USS_VERSION_DYNAMIC, data_.uss_version);
    }
    DeviceInfo GetData() {
        std::lock_guard<std::mutex> lock(mutex);
        return data_;
    }

    void SetMajorVersion(std::string &version) {
        std::lock_guard<std::mutex> lock(mutex);
        data_.major_version = version;
    }
    void SetMcuVersion(std::string &version) {
        std::lock_guard<std::mutex> lock(mutex);
        data_.mcu_version = version;
    }
    void SetSocVersion(std::string &version) {
        std::lock_guard<std::mutex> lock(mutex);
        data_.soc_version = version;
    }
    void SetSwtVersion(std::string &version) {
        std::lock_guard<std::mutex> lock(mutex);
        data_.swt_version = version;
    }
    void SetDsvVersion(std::string &version) {
        std::lock_guard<std::mutex> lock(mutex);
        data_.dsv_version = version;
    }
    void SetUssVersion(std::string &version) {
        std::lock_guard<std::mutex> lock(mutex);
        data_.uss_version = version;
    }

    void SetDeviceInfo(DeviceInfo &data) {
        std::lock_guard<std::mutex> lock(mutex);
        data_ = data;
    }
private:
    DeviceInfomation(){}
    ~DeviceInfomation(){}

    std::mutex mutex;
    DeviceInfo data_{};
};

class DeviceStatus {

public:
    static DeviceStatus* getInstance() {
        static DeviceStatus instance;
        return &instance;
    }
    Devicestatus GetData() {
        std::lock_guard<std::mutex> lock(mutex);
        return data_;
    }
    void SetDeviceStatus(Devicestatus &data) {
        std::lock_guard<std::mutex> lock(mutex);
        data_ = data;
    }
private:
    DeviceStatus(){}
    ~DeviceStatus(){}

    std::mutex mutex;
    Devicestatus data_{};
};

class TemperatureDataInfo {
public:
    static TemperatureDataInfo* getInstance() {
        static TemperatureDataInfo instance;
        return &instance;
    }
    TemperatureData GetData() {
        std::lock_guard<std::mutex> lock(mutex);
        return data_;
    }
    void SetTemperature(TemperatureData &data) {
        std::lock_guard<std::mutex> lock(mutex);
        data_ = data;
    }
private:
    TemperatureDataInfo(){}
    ~TemperatureDataInfo(){}

    std::mutex mutex;
    TemperatureData data_;
};

class VoltageDataInfo {
public:
    static VoltageDataInfo* getInstance() {
        static VoltageDataInfo instance;
        return &instance;
    }
    VoltageData GetData() {
        std::lock_guard<std::mutex> lock(mutex);
        return data_;
    }
    void SetVoltage(VoltageData &data) {
        std::lock_guard<std::mutex> lock(mutex);
        data_ = data;
    }
private:
    VoltageDataInfo(){}
    ~VoltageDataInfo(){}

    std::mutex mutex;
    VoltageData data_;
};

}  // namespace devm_server
}  // namespace netaos
}  // namespace hozon


