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
 * @file NCErrorPubDef.h
 * @brief
 * @date 2020-05-09
 *
 */

#ifndef NCERRORPUBDEF_H
#define NCERRORPUBDEF_H

#include "osal/ncore/NCTypesDefine.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE
const CHAR* const NC_ERROR_MSG_CMD_ERRORLOG_CLIENT    = "errlogClient";
const CHAR* const NC_ERROR_MSG_CMD_GETNUM_CLIENT      = "getNumClient";
const CHAR* const NC_ERROR_MSG_CMD_CLOSEERRDFD_CLIENT = "closeErrdFd";

const UINT32 NC_ERROR_MSG_CMD_MAX_SIZE = 16;

const UINT32 NC_ERROR_SINGLE_LOG_SIZE = 1024;  // a single log size
const UINT32 NC_ERROR_TIME_BUF_SIZE   = 32;

const UINT32 NC_ERROR_FILE_PATH_SIZE = 2048;
const UINT32 NC_ERROR_FILE_NAME_SIZE = 32;
const UINT32 NC_ERROR_EXT_NAME_SIZE  = 32;
const UINT32 NC_ERROR_VERSION_SIZE   = 16;

const CHAR* const NC_ERROR_VERSION = "Ver1.1";

/**
 * NC Error type definition
 */
enum NC_ERRORTYPE {
    NC_ERROR_DEBUG = 0,  // Debug Error
    NC_ERROR_ERROR,      // Normal error
    NC_ERROR_FATAL,      // fatal error
    NC_ERROR_LEVEL_NUM   // error level number
};

// NCError Header section
struct NCErrorMsgHeader {
    CHAR  cmd[ NC_ERROR_MSG_CMD_MAX_SIZE ];
    INT32 dataSize;
};

// msg buffer
struct NCErrorLogFormat {
    INT32 msgType;                              // msg type
    CHAR  logTime[ NC_ERROR_TIME_BUF_SIZE ];    // Time
    CHAR  msgText[ NC_ERROR_SINGLE_LOG_SIZE ];  // msg data buf
};

// NCErrorLog info Clinet
struct NCErrorLogInfoClient {
    NCErrorMsgHeader errlogHeaderClient;
    NCErrorLogFormat errlog;
};

// Values sended by Server
union ValuesFromServer {
    INT32 nums[ NC_ERROR_LEVEL_NUM ];
    INT32 flag;
};
OSAL_END_NAMESPACE

#endif /* NCERRORPUBDEF_H */
/* EOF */
