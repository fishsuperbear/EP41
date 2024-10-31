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
 * @file NCWaitObj.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCWAITOBJ_H_
#define INCLUDE_NCORE_NCWAITOBJ_H_

#include <pthread.h>

#include "osal/ncore/NCTypesDefine.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE

// Class declaration
class NCWaitObj;

/**
 * @brief
 *
 * @class NCWaitObj
 */
class __attribute__( ( visibility( "default" ) ) ) NCWaitObj {
   public:
    /**
     * Construction.
     *
     * @param manual
     * - NC_TRUE : Creates a manual-reset event object
     * - NC_FALSE : Creates an auto-reset event object
     */
    explicit NCWaitObj( NC_BOOL manual = NC_FALSE );

    /**
     * Destruction.
     */
    virtual ~NCWaitObj();

    /**
     * wait for single objects
     *
     * @param msec : Time-out interval, in milliseconds.
     *
     * @return NC_BOOL : NC_TRUE means succeed, and NC_FALSE failed.
     */
    NC_BOOL waitTime( UINT32 msec = INFINITE32 );

    /**
     * Set the specified event object to the signaled state.
     */
    VOID notify();

    /**
     * Set the specified event object to the nonsignaled state
     */
    VOID reset();

   private:
    pthread_mutex_t    m_mutex;
    pthread_cond_t     m_cond;
    pthread_condattr_t m_condattr;
    NC_BOOL            signal_flag;
    NC_BOOL            manual_flag;

   private:
    NCWaitObj( const NCWaitObj &src );
    NCWaitObj &operator=( const NCWaitObj &src );
};

OSAL_END_NAMESPACE
#endif  // INCLUDE_NCORE_NCWAITOBJ_H_
/* EOF */
