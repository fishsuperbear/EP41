/**
 * * --------------------------------------------------------------------
 * * |                                                                  |
 * * |     _         _    _ _______ ____         _____ ____  __  __     |
 * * |    (_)   /\  | |  | |__   __/ __ \       / ____/ __ \|  \/  |    |
 * * |     _   /  \ | |  | |  | | | |  | |     | |   | |  | | \  / |    |
 * * |    | | / /\ \| |  | |  | | | |  | |     | |   | |  | | |\/| |    |
 * * |    | |/ ____ \ |__| |  | | | |__| |  _  | |___| |__| | |  | |    |
 * * |    |_/_/    \_\____/   |_|  \____/  (_)  \_____\____/|_|  |_|    |
 * * |                                                                  |
 * * --------------------------------------------------------------------
 *
 *  * Copyright @ 2020 iAuto (Shanghai) Co., Ltd.
 *  * All Rights Reserved.
 *  *
 *  * Redistribution and use in source and binary forms, with or without
 *  * modification, are NOT permitted except as agreed by
 *  * iAuto (Shanghai) Co., Ltd.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS,
 *  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 *
 * @file dlt_ctrl_func_def.h
 * @brief
 * @date 2020-08-10
 *
 */
#ifndef DLT_CTRL_FUNC_DEF_H
#define DLT_CTRL_FUNC_DEF_H

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include <string>
#include <vector>
#include "osal/dlt/dlt_protocol.h"

namespace dlt {
typedef struct {
    uint8_t applicationid[ 4 ];
    uint8_t contextid[ 4 ];
    int8_t  newLogLevel;
    uint8_t reserved[ 4 ];
} PACKED SetLogLevel_IN;

typedef struct {
    uint8_t options;
    uint8_t applicationId[ 4 ];
    uint8_t contextId[ 4 ];
    uint8_t reserved[ 4 ];
} PACKED GetLogInfo_IN;

typedef struct {
    typedef struct {
        uint8_t     contextId[ 4 ];
        uint8_t     logLevel;
        uint8_t     traceStatus;
        std::string contextDesc;
    } contextIdInfo;

    typedef struct {
        uint8_t                    appID[ 4 ];
        std::vector<contextIdInfo> contextIdInfoList;
        std::string                appDesc;
    } appIdInfo;
    std::vector<appIdInfo> appIdInfoList;
} GetLogInfo_OUT_LogInfoType;

typedef struct {
    int8_t  newLogLevel;
    uint8_t reserved[ 4 ];
} PACKED SetDefaultLogLevel_IN;

typedef struct {
    std::string module;
    uint64_t    pid;
} DumpStatistic_IN;
}
#endif