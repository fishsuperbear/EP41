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
 * @file NCTimerThreadPool.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCTIMERTHREADPOOL_H_
#define INCLUDE_NCORE_NCTIMERTHREADPOOL_H_

#if ( defined __QNX__ ) || ( defined __QNXNTO__ )
#include <sys/dispatch.h>
#endif
#include <map>

#include "osal/ncore/NCList.h"
#include "osal/ncore/NCSyncObj.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE

class NCTimer;

/**
 * @brief
 *
 * @class NCTimerThreadPool
 */
class NCTimerThreadPool : public NCSyncObj {
   public:
    NCTimerThreadPool() {}
    virtual ~NCTimerThreadPool() {}

    VOID    startTimerThreadPool() {}
    VOID    stopTimerThreadPool() {}
    NC_BOOL registerTimer( NCTimer *timer ) { return NC_TRUE; }
    NC_BOOL stopTimer( NCTimer *timer ) { return NC_TRUE; }
    NC_BOOL restartTimer( NCTimer *timer ) { return NC_TRUE; }

    NC_BOOL removeTimer( NCTimer *timer ) { return NC_TRUE; }

    NC_BOOL isValid( INT32 id ) { return NC_TRUE; }

   private:
    NCTimerThreadPool( const NCTimerThreadPool & );
    NCTimerThreadPool &operator=( const NCTimerThreadPool & );
};

OSAL_END_NAMESPACE

#endif  // INCLUDE_NCORE_NCTIMERTHREADPOOL_H_
/* EOF */
