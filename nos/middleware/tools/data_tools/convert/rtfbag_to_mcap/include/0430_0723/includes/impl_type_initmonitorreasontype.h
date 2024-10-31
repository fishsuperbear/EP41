/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef IMPL_TYPE_INITMONITORREASONTYPE_H
#define IMPL_TYPE_INITMONITORREASONTYPE_H

#include "impl_type_uint8.h"
enum class InitMonitorReasonType : UInt8
{
    kClear = 0,
    kRestart = 1,
    kReenabled = 2,
    kStorageEnabled = 3
};


#endif // IMPL_TYPE_INITMONITORREASONTYPE_H
