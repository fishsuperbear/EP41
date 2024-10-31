/*
 * @Author: Z04975 liguoqiang@hozonauto.com
 * @Date: 2023-11-22 14:14:22
 * @LastEditors: Z04975 liguoqiang@hozonauto.com
 * @LastEditTime: 2023-12-06 16:25:36
 * @FilePath: /lgq/nos/service/config_server/include/cfg_data_def.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */

/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: cfg
 * Description: 配置管理数据类型定义
 * Created on: Feb 7, 2023
 *
 */
#ifndef SERVICE_CONFIG_SERVER_INCLUDE_CFG_DATA_DEF_H_
#define SERVICE_CONFIG_SERVER_INCLUDE_CFG_DATA_DEF_H_
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace hozon {
namespace netaos {
namespace cfg {
using namespace std;
enum DATA_TYPE : uint8_t {
    CFG_DATA_OTHER = 0,
    CFG_DATA_BOOL,
    CFG_DATA_DOUBLE,
    CFG_DATA_FLOAT,
    CFG_DATA_INT32,
    CFG_DATA_UINT8,
    CFG_DATA_LONG,
    CFG_DATA_STRING,
    CFG_DATA_VECTOR_UINT8,
    CFG_DATA_VECTOR_BOOL,
    CFG_DATA_VECTOR_DOUBLE,
    CFG_DATA_VECTOR_FLOAT,
    CFG_DATA_VECTOR_INT32,
    CFG_DATA_VECTOR_LONG,
    CFG_DATA_VECTOR_STRING
};
enum CfgResultCode {
    CONFIG_TIME_OUT = 0,
    CONFIG_OK,
    CONFIG_ERROR,
    CONFIG_SERVICE_NO_READY,
    CONFIG_NO_VALUE,
    CONFIG_OVER_RANGE_VALUE,
    CONFIG_TYPE_MISMATCH_PARAM_VALUE,
    CONFIG_INVALID_PARAM,
    CONFIG_INVALID_PARAM_VALUE,
    CONFIG_INVALID_WAIT_TIME,
    CONFIG_NO_PARAM,
    CONFIG_GET_CHIPID_FAILED,
    CONFIG_FILE_RW_FAILED,
    CONFIG_UNSUPPORT
};

enum class ServerNotifyType : int32_t { REPORT_ALIVE = 0, REPORT_PARAMS };
enum ConfigPersistType : std::uint8_t { CONFIG_NO_PERSIST, CONFIG_ASYNC_PERSIST, CONFIG_SYNC_PERSIST };
enum ConfigRefType : std::uint8_t { CONFIG_KEY_VALUE, CONFIG_REF_KEY_VALUE, CONFIG_REF_FILE };
enum ConfigMethodType : std::uint8_t {
    CONFIG_INITCLIENT,
    CONFIG_DEINITCLIENT,
    CONFIG_SETPARAM,
    CONFIG_GETPARAM,
    CONFIG_RESETPARAM,
    CONFIG_DELPARAM,
    CONFIG_MONITORPARAM,
    CONFIG_UNMONITORPARAM,
    CONFIG_PARAMUPDATEDATARES,
    CONFIG_GETMONITORCLIENTS,
    CONFIG_GETCLIENTINFOLIST
};
enum CfgUpdateToMcuFlag : uint8_t { NOTIFY_STOP = 0, NOTIFY_PEDDING, NOTIFY_UPDATING };
const std::string allkey = "all";
const std::string vehiclecfgveckey = "dids/F170";
const std::string vehiclecfgkey = "vehiclecfg/";

}  // namespace cfg
}  // namespace netaos
}  // namespace hozon
#endif  // SERVICE_CONFIG_SERVER_INCLUDE_CFG_DATA_DEF_H_
