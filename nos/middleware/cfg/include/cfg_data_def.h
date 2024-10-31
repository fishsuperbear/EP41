

/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: cfg
 * Description: 配置管理数据类型定义
 * Created on: Feb 7, 2023
 *
 */
#ifndef MIDDLEWARE_CFG_INCLUDE_CFG_DATA_DEF_H_
#define MIDDLEWARE_CFG_INCLUDE_CFG_DATA_DEF_H_
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace hozon {
namespace netaos {
namespace cfg {
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
    CONFIG_UNSUPPORT,
    CONFIG_SERIALIZE_FAILED,
    CONFIG_DESERIALIZE_FAILED
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
    CONFIG_PARAMUPDATEDATAEVENT,
    CONFIG_PARAMUPDATEDATARESEVENT,
    CONFIG_GETPARAMINFOLIST,
    CONFIG_GETCLIENTINFOLIST
};

struct CfgParamInfo {
    std::string param_name;
    int32_t data_type;
    int32_t data_size;
    std::string lastupdate_clientname;
    std::string lastupdate_time;
    int32_t per_plag;
};
struct CfgMethodEventInfo {
    int32_t type;
    std::string param_name;
    int32_t count;
};

struct CfgClientInfo {
    std::string client_name;
    std::vector<std::string> monitor_params;
    std::vector<CfgMethodEventInfo> methodevent_info;
    int32_t running_status;
    std::string start_time;
    std::string end_time;
};

enum CmServiceStatus : std::uint8_t { CONFIG_UNINIT, CONFIG_INITERROR, CONFIG_INITSUCC };
const bool protoflag = false;
const uint32_t VALUE_MAX_SIZE = 10 * 1024 * 1024U;
const uint32_t KEY_MAX_SIZE = 128U;
const uint32_t WAIT_MAX_TIMEOUT = 20000;
const uint32_t WAIT_MIX_TIMEOUT = 1000;

}  // namespace cfg
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_CFG_INCLUDE_CFG_DATA_DEF_H_
