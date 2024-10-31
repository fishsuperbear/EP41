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
 * @file NCThreadKey.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCTHREADKEY_H_
#define INCLUDE_NCORE_NCTHREADKEY_H_

#include <pthread.h>

#include "osal/ncore/NCSyncObj.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE

// Class declaration
class NCThreadKey;

/**
 * @brief
 *
 * @class NCThreadKey
 */
class __attribute__( ( visibility( "default" ) ) ) NCThreadKey {
   public:
    /**
     * Get the NCThreadKey's instance. If it not exist, create it.
     *
     * @return the instance.
     */
    static NCThreadKey *Instance() EXCLUDES( s_cSync );

    /**
     * Destroy the instance.
     */
    static VOID Destory() EXCLUDES( s_cSync );

    /**
     * Set priority to current thread.
     *
     * @param  Priority: new priority.
     */
    static VOID setCurrentThreadPriority( const INT32 Priority );

    /**
     * Get current thread priority.
     *
     * @return  current priority.
     */
    static INT32 getCurrentThreadPriority();

    /**
     * Get the thread object address where the thread key created.
     *
     * @return  the private data.
     */
    VOID *getThread() const;

    /**
     * Associate with thread's private data and the thread's key.
     *
     * @param  thread's private data
     */
    VOID setThread( const VOID *const pThread ) const;

   private:
    pthread_key_t                   m_key;
    static NCThreadKey *s_pInstance GUARDED_BY( s_cSync );
    static NCSyncObj                s_cSync;

   private:
    NCThreadKey();
    virtual ~NCThreadKey();

    NCThreadKey( const NCThreadKey & );
    NCThreadKey &operator=( const NCThreadKey & );
};
OSAL_END_NAMESPACE
#endif  // INCLUDE_NCORE_NCTHREADKEY_H_
/* EOF */
