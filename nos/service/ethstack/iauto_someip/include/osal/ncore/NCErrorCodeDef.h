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
 * @file NCErrorCodeDef.h
 * @brief
 * @date 2020-05-09
 *
 */

#ifndef NCERRORCODEDEF_H
#define NCERRORCODEDEF_H

#ifndef NCTYPESDEFINE_H
#include "osal/ncore/NCTypesDefine.h"
#endif

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE
/**
 * error code format:
 * Module:28bit~31bit
 * Sub Module:20bit~27bit
 * error number:0bit~19bit
 * example:0x002000ab, module is NC_ERRORCODE_MODULE_SYSTEM,
 * sub module is NC_ERRORCODE_SYSTEM_SUB_MODULE_CORE,
 * error number is 0xab.
 */
enum NC_ERRORCODE_MODULE {
    NC_ERRORCODE_MODULE_SYSTEM   = 0,  // 0x00000000~0x0FFFFFFF
    NC_ERRORCODE_MODULE_PLATFORM = 1,  // 0x10000000~0x1FFFFFFF
    NC_ERRORCODE_MODULE_FRAMWORK = 2,  // 0x20000000~0x2FFFFFFF
    NC_ERRORCODE_MODULE_HAL      = 3,  // 0x30000000~0x3FFFFFFF
    NC_ERRORCODE_MODULE_NUMS
};

enum NC_ERRORCODE_SYSTEM_SUB_MODULE {
    NC_ERRORCODE_SYSTEM_SUB_MODULE_RESERVE = 0,  // 0x00000000~0x000FFFFF
    NC_ERRORCODE_SYSTEM_SUB_MODULE_CORE,         // 0x00100000~0x001FFFFF (ncore)
    NC_ERRORCODE_SYSTEM_SUB_MODULE_HANDLER,      // 0x00200000~0x002FFFFF (device manager)
    NC_ERRORCODE_SYSTEM_SUB_MODULE_TESTMODE,     // 0x00300000~0x003FFFFF (test mode)
    NC_ERRORCODE_SYSTEM_SUB_MODULE_UPDATE,       // 0x00400000~0x004FFFFF (system update)
    NC_ERRORCODE_SYSTEM_SUB_MODULE_SHAREDDATA,   // 0x00500000~0x005FFFFF (sharedmemorydata)
    NC_ERRORCODE_SYSTEM_SUB_MODULE_CHECKFILE,    // 0x00600000~0x006FFFFF (NCCheckFile)
    NC_ERRORCODE_SYSTEM_SUB_MODULE_WAYLAND,      // 0x00700000~0x007FFFFF (wayland)
    NC_ERRORCODE_SYSTEM_SUB_MODULE_EVSYS,        // 0x00800000~0x008FFFFF (eventsys)
    NC_ERRORCODE_SYSTEM_SUB_MODULE_NUMS
};

enum NC_ERRORCODE_FRAMWORK_SUB_MODULE {
    NC_ERRORCODE_FRAMWORK_SUB_MODULE_EVENT,  // 0x00000000~0x000FFFFF (event)
    NC_ERRORCODE_FRAMWORK_SUB_MODULE_NUMS
};

// Convert to a error code
#define NCERRORCODE( module, subModule, error ) \
    ( ( module ) << 28 | ( subModule ) << 20 | ( error ) )
// Convert to a system module error code
#define NCERRORCODES( subModule, error ) ( 0x00000000 | ( subModule ) << 20 | ( error ) )
// Convert to a platform module error code
#define NCERRORCODEP( subModule, error ) ( 0x10000000 | ( subModule ) << 20 | ( error ) )
// Convert to a framwork module error code
#define NCERRORCODEF( subModule, error ) ( 0x20000000 | ( subModule ) << 20 | ( error ) )
// Convert to a hal module error code
#define NCERRORCODEH( subModule, error ) ( 0x30000000 | ( subModule ) << 20 | ( error ) )

// get the module
#define NCERRORCODE_MODULE( errorCode ) ( ( errorCode ) >> 28 )
// get the sub module
#define NCERRORCODE_SUB_MODULE( errorCode ) ( ( errorCode ) >> 20 & 0x0FF )
// get the error numbers
#define NCERRORCODE_ERROR( errorCode ) ( (errorCode) &0x000FFFFF )
OSAL_END_NAMESPACE

#endif /* NCERRORCODEDEF_H */
/* EOF */
