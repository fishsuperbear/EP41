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
 * @file NCThreadSystemIF.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCTHREADSYSTEMIF_H_
#define INCLUDE_NCORE_NCTHREADSYSTEMIF_H_

#include "osal/ncore/NCTypesDefine.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE
#if ( defined __QNX__ ) || ( defined __QNXNTO__ )
static const size_t NC_THREAD_INFO_NAME_LEN = 100U;
#elif ( defined __linux__ ) || ( defined linux )
static const size_t NC_THREAD_INFO_NAME_LEN = 16U;
#else
#error Not support in the system.
#endif
static const UINT32 NC_THREAD_PRIORITY_IDX_NORMAL = 0U;
static const UINT32 NC_THREAD_PRIORITY_IDX_LOW    = 1U;
static const UINT32 NC_THREAD_PRIORITY_IDX_HIGH   = 2U;

// Thread Table
struct NC_THREAD_TABLE {
    const CHAR *thread_name;    // thread name
    INT32       priority[ 3 ];  // thread priority,
                                // include {normal, low, high}
    UINT32 sanity_interval;     // sanity check(s)
};

struct NCThreadSystemIF {
    virtual ~NCThreadSystemIF();

    virtual NC_BOOL getThreadTableInfo( const CHAR *name, UINT32 *const priority,
                                        UINT32 &sanity ) = 0;

    virtual NC_BOOL addThreadTable( const NC_THREAD_TABLE *const ) = 0;

    // return thread No.
    virtual INT32 registerThread( const CHAR *const name, const VOID *const handle,
                                  UINT32 thread_id ) = 0;

    virtual VOID unregisterThread( INT32 thread_no ) = 0;
};

__attribute__( ( visibility( "default" ) ) ) VOID NC_SetThreadSystem( NCThreadSystemIF *const sp );
__attribute__( ( visibility( "default" ) ) ) NCThreadSystemIF *NC_GetThreadSystem();
OSAL_END_NAMESPACE
#endif  // INCLUDE_NCORE_NCTHREADSYSTEMIF_H_
/* EOF */
