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
 * @file NCUseCounter.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCUSECOUNTER_H_
#define INCLUDE_NCORE_NCUSECOUNTER_H_

#include <pthread.h>

#include "osal/ncore/NCTypesDefine.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE

// Class declaration
class NCUseCounter;

/**
 * @brief
 *
 * @class NCUseCounter
 */
class __attribute__( ( visibility( "default" ) ) ) NCUseCounter {
   public:
    /**
     * Construction.
     */
    NCUseCounter();

    /**
     * Destruction.
     */
    virtual ~NCUseCounter();

    /**
     * Lock
     *
     * @param msec : The time of locking
     *
     * @return NC_BOOL : NC_TRUE means lock succeed, and NC_FALSE failed.
     */
    NC_BOOL Lock( UINT64 msec = INFINITE64 );

    /**
     * Unlock.
     */
    VOID Unlock();

    /**
     * Reference count increase.
     */
    VOID Use();

    /**
     * Reference count decrease.
     */
    VOID Unuse();

    /**
     * Get the use state.
     *
     * @return NC_BOOL
     * - NC_TRUE : Using
     * - NC_FALSE : Unused
     */
    NC_BOOL IsUsed() const;

   private:
    INT32               use_count;
    pthread_mutex_t     m_mutex;
    pthread_mutexattr_t m_attr;

   private:
    NCUseCounter( const NCUseCounter &src );
    NCUseCounter &operator=( const NCUseCounter &src );
};

OSAL_END_NAMESPACE
#endif  // INCLUDE_NCORE_NCUSECOUNTER_H_
/* EOF */
