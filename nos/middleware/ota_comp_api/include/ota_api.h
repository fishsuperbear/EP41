
/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: ota api definition
 */

#ifndef HOZON_OTA_API_H_
#define HOZON_OTA_API_H_

#include <stdint.h>


#define API_VERSION "V00.01.00"
enum STD_RTYPE_E {
    E_OK,
    E_NOT_OK,
};

//升级状态
enum OTA_UPDATE_STATUS_E {
    OTA_UPDATE_STATUS_IDLE,         //空闲状态
    OTA_UPDATE_STATUS_PROCESSING,   //正在进行升级
    OTA_UPDATE_STATUS_SUCCESS,      //升级成功
    OTA_UPDATE_STATUS_FAIL,         //升级失败
};

enum OTA_LOG_TYPE {
    OTA_DEBUG_LOG,
    OTA_INFO_LOG,
    OTA_WARNING_LOG,
    OTA_ERROR_LOG,
    OTA_FATAL_LOG,
};
typedef void (*OTA_LOG_HANDLER)(enum OTA_LOG_TYPE type, const uint8_t* log);

enum STD_RTYPE_E ota_api_init();
enum STD_RTYPE_E ota_api_deinit();
enum STD_RTYPE_E ota_get_api_version(int8_t * api_version);
enum STD_RTYPE_E ota_start_update(uint8_t * file_path);
enum STD_RTYPE_E ota_get_update_status(enum OTA_UPDATE_STATUS_E * ota_update_status, uint8_t * progress);
enum STD_RTYPE_E ota_get_log_path (uint8_t *log_path);
enum STD_RTYPE_E ota_log_callback_register (OTA_LOG_HANDLER ota_log_handler);


#endif
