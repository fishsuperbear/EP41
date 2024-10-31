/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: 配置管理数据类型定义
 * Create: 2020-03-26
 */
#ifndef CONFIG_DATA_DEFINE_H
#define CONFIG_DATA_DEFINE_H

#include "base_type.h"
#include <functional>

namespace mdc {
namespace config {

enum ConfigResultCode {
    CONFIG_OK = 0,
    CONFIG_ERROR,
    CONFIG_SERVICE_NO_READY,
    CONFIG_NO_VALUE,
    CONFIG_TIME_OUT,
    CONFIG_INVALID_PARAM,
    CONFIG_INVALID_PARAM_VALUE,
    CONFIG_INVALID_WAIT_TIME,
    CONFIG_PARAM_NOT_EXIST,
    CONFIG_GET_CHIPID_FAILED,
    CONFIG_FILE_RW_FAILED,
    CONFIG_UNSUPPORT
};

enum class ServerNotifyType : int32_t {
    REPORT_ALIVE = 0,
    REPORT_PARAMS
};

enum class ParamType : std::uint8_t {
    TYPE_INT,
    TYPE_UINT,
    TYPE_UINT8,
    TYPE_DOUBLE,
    TYPE_STRING,
    TYPE_BOOLEAN,
    TYPE_VEC_UINT8,
    TYPE_UNKNOW
};

enum ConfigPersistType : std::uint8_t {
    CONFIG_NO_PERSIST,
    CONFIG_ASYNC_PERSIST,
    CONFIG_SYNC_PERSIST
};
}
}
#endif