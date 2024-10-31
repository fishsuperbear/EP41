/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: data definition
 */
#ifndef OTA_DATA_DEF_H
#define OTA_DATA_DEF_H

#include <stdint.h>
#include <memory>
#include <map>
#include <vector>
#include <string>
#include <thread>

namespace hozon {
namespace netaos {
namespace update {

using namespace std;

const uint16_t UPDATE_MANAGER_FUNCTIONAL_ADDR_DOCAN     = 0x7DF;
const uint16_t UPDATE_MANAGER_FUNCTIONAL_ADDR_DOIP      = 0xE400;

const uint16_t UDS_SID_22        = 0x22;
const uint16_t UDS_SID_2E        = 0x2E;
const uint16_t UDS_SID_31        = 0x31;
const uint16_t UDS_SID_38        = 0x38;
const uint16_t UDS_SID_11        = 0x11;
const uint16_t UDS_SID_28        = 0x28;

const uint16_t UDS_METHOD_RES_ACK  = 0;
const uint16_t UDS_METHOD_RES_NACK = 1;

const std::string UPDATE_MODE_OTA = "Update";
const std::string UPDATE_MODE_NORMAL = "Normal";

const std::string STATE_FILE_PATH = "/cfg/conf_ota/";
const std::string UM_STATE_FILE = "um_state";

const std::string VERSION_FILE_PATH = "/app/version.json";

const std::string SENSOR_LIDAR = "LIDAR";
const std::string SENSOR_SRR_FL = "SRR_FL";
const std::string SENSOR_SRR_FR = "SRR_FR";
const std::string SENSOR_SRR_RL = "SRR_RL";
const std::string SENSOR_SRR_RR = "SRR_RR";

const std::string aes_key = "MIICXQIBAAKBgODVMIICXQIBAAKBgQDV";

const std::string OTA_RESULT_FILE_PATH = "/opt/usr/log/ota_log/ota_result.json";

const std::string CFG_DIDS_FILE = "/cfg/dids/dids.json";
const std::string CFG_VERSION_FILE = "/cfg/version/version.json";

const std::string CMD_FLAG_FILE = "/cfg/ota/ota.json";

const uint16_t max_retries = 3;
const std::string UPDATE_SERVICE = "svp_update";
typedef struct uds_data_req
{
    uds_data_req():
    sid(0),
    subid(0),
    resp_ack(0),
    data_len(0)
    {}
    map<string, string> meta_info; // sa, ta, ta_type, ip, port, url
    uint16_t sid;
    uint16_t subid;
    uint16_t resp_ack;
    uint32_t data_len;
    vector<uint8_t> data_vec;
} uds_data_req_t;

typedef struct uds_data_resp
{
    uint16_t sid;
    uint16_t subid;
    uint8_t res_code;
} uds_data_resp_t;

typedef struct uds_raw_data_req
{
    uint16_t sa;
    uint16_t ta;
    uint8_t bus_type;
    uint32_t data_len;
    vector<uint8_t> data_vec;
} uds_raw_data_req_t;

typedef struct uds_raw_data_resp
{
    uint16_t sa;
    uint16_t ta;
    uint8_t bus_type;
    uint8_t result;
    vector<uint8_t> data_vec;
} uds_raw_data_resp_t;


typedef struct chassis_info
{
    chassis_info() : gear_display{0}, vehicle_speed_vaid{false}, vehicle_speed{0} {}
    uint8_t gear_display;
    bool vehicle_speed_vaid;
    float vehicle_speed;
} chassis_info_t;

enum class DiagUpdateStatus {
    kDefault = 0x00,
    kUpdating = 0x01,
    kUpdated = 0x02
};

enum UpdateStatus {
    kIdle = 0,
    kReady = 1,
    kBusy = 2,
    kActivating = 3,
    kActivated = 4,
    kRollingBack = 5, // not support current
    kRolledBack = 6,  // not support current
    kCleaning_Up = 7,
    kVerifying = 8,
    kServiceNotAvailable = 9,
};

enum installer_type {
    INSTALLER_DOCAN = 1,
    INSTALLER_DOIP = 2
};

enum DownloadResult {
    DOWNLOAD_SUCCESS           =  0,
    DOWNLOAD_RECOVER           =  1,
    DOWNLOAD_MD5CHECK_SUCCESS  =  2,
    DOWNLOAD_VERIFY_SUCCESS    =  3,
    DOWNLOAD_HASHCHECK_SUCCESS =  4,

    DOWNLOAD_INTERRUPT         = -1,
    DOWNLOAD_SPACE_NOT_ENOUGH  = -2,
    DOWNLOAD_MD5CHECK_FAILED   = -3,
    DOWNLOAD_VERIFY_FAILED     = -4,
    DOWNLOAD_HASHCHECK_FAILED  = -5
};

typedef struct download_status
{
    int32_t downloadResult;
    uint32_t downloadProgress;
} download_status_t;

typedef struct update_item_state
{
    /* data */
    std::string name;
    uint8_t     progress;
    int32_t     errorCode;
    std::string message;
}update_item_state_t;

struct SensorsDataInfo {
    std::string sensorName;
    uint8_t busType;
    uint16_t logicAddr;
};

enum UpdateManagerInfoDataType {
    kHEX = 0x01,
    kBCD = 0x02,
    kASCII = 0x03
};

const std::string mcu_uds_ip = "172.16.90.11";
const uint32_t mcu_uds_port = 23460;

using McuUdsMsg = std::array<uint8_t, 4>;

}  // namespace update
}  // namespace netaos
}  // namespace hozon
#endif  // OTA_DATA_DEF_H
