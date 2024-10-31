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
 * @file dlt_ctrl_func.h
 * @brief
 * @date 2020-06-30
 *
 */
#ifndef DLT_CTRL_FUNC_H
#define DLT_CTRL_FUNC_H

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include <stdint.h>
#include <stdlib.h>

#include <functional>
#include <string>

#include "osal/dlt/dlt_common.h"
#include "osal/dlt/dlt_conf.h"
#include "osal/dlt/dlt_ctrl.h"
#include "osal/dlt/dlt_ctrl_func_def.h"
#include "osal/dlt/dlt_payload.h"
#include "osal/dlt/dlt_protocol.h"

namespace dlt {
class DltLogItemArg {
   public:
    DltLogItemArg( const DltLogItem *item, int32_t fd );
    ~DltLogItemArg();

   public:
    const DltLogItem *mItem;
    int32_t           mfd;
};

class DltLogItemPackerArg {
   public:
    DltLogItemPackerArg( const DltCtrlMsgPacker *packer, int32_t fd );
    ~DltLogItemPackerArg();

   public:
    const DltCtrlMsgPacker *mPacker;
    int32_t                 mfd;
};

class DLTCtlMethord {
   public:
    DLTCtlMethord();
    ~DLTCtlMethord();

    virtual bool SetLogLevel( const SetLogLevel_IN &in, uint8_t &out_status );
    virtual bool StoreConfiguration( uint8_t &out_status );
    virtual bool GetLogInfo( const GetLogInfo_IN &in, uint8_t &out_status,
                             GetLogInfo_OUT_LogInfoType &out );
    virtual bool GetDefaultLogLevel( uint8_t &out_status, uint8_t &loglevel );
    virtual bool ResetToFactoryDefault( uint8_t &out_status );
    virtual bool SetMessageFiltering( uint8_t newstatus, uint8_t &out_status );
    virtual bool SetDefaultLogLevel( const SetDefaultLogLevel_IN &in, uint8_t &out_status );
    virtual bool GetSoftwareVersion( uint8_t &out_status, std::string &swversion );
    // userdefined
    virtual bool SetFilter( const std::vector<std::string> &filter, uint8_t &out_status );
    virtual bool HandleCfgInfo( const std::string &info, const uint8_t &isAdd,
                                uint8_t &out_status );
    virtual bool ChangeMode( const std::string &apid, uint8_t &out_status );
    virtual bool DumpStatistic( const DumpStatistic_IN &info, uint8_t &out_ustatus,
                                std::string &str_result );
    virtual bool ClearCache( const std::string &buffer, uint8_t &out_status );
};

class DLTCtlFuncClient {
   public:
    // receive ctrl func
    static void OnCtrlPackageReceive( const DltLogItemArg &item );

    // on ctrl msg packed over
    static bool RegeditCtrlMsgPackerCallBack(
        std::function<bool( const DltLogItemPackerArg &item )> func );

    // methord func
    static DLTCtlMethord *GetCtrlMethordInstance();
};

class DLTCtlFuncServer {
   public:
    // receive ctrl func
    static void OnCtrlPackageReceive( const DltLogItemArg &itemArg );

    // on ctrl msg packed
    static bool RegeditCtrlMsgPackerCallBack(
        std::function<bool( const DltLogItemPackerArg &item )> func );

    // on methord func called
    static bool SetCtrlMethordInstance( DLTCtlMethord *ins );
};
}  // namespace dlt
#endif /* __DLT_CTRL_FUNC_H__ */
