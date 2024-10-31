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
 * @file dlt.h
 * @brief
 * @date 2020-06-18
 *
 */
#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#ifndef DLT_H
#define DLT_H
#include "osal/dlt/dlt_conf.h"
#include "osal/dlt/dlt_ctrl.h"
#include "osal/dlt/dlt_ctrl_func.h"
#include "osal/dlt/dlt_log.h"

namespace dlt {
class DltLogPacker {
   public:
    static bool Init( const DltLogCfgInfo *info );
    static DltLogItemPacker *CreateLogItemPacker( const std::string &tag, DltLogLevelType level,
                                                  const DltLogChannal &channal );

    static DltCtrlMsgPacker *CreateCtrlMsgPacker();

    static DLTStorageHeader &GetDLTStorageHeader( const std::string &ecu );
};

class DltLogParse {
   public:
    static bool ParseMsgBuf( const uint8_t *buffer, uint32_t len, DltLogItem *logitem );
    static DltPayload *ParseMsgPayLoad( const uint8_t *buffer, uint32_t len );

    static std::string ParseMsgToString( const DltLogItem *item, bool useColor = true );
};
}  // namespace dlt
#endif