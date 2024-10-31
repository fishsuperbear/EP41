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
 * @file NCThreadQueueWpKey.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCTHREADQUEUEWPKEY_H_
#define INCLUDE_NCORE_NCTHREADQUEUEWPKEY_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <pthread.h>

#include "osal/ncore/NCGlobalAPI.h"
#include "osal/ncore/NCSyncObj.h"
#include "osal/ncore/NCTypesDefine.h"

OSAL_BEGIN_NAMESPACE
//  for called when thread quit to release NCRunnableQueueWeakHolder
//  in the thread-specific data area
void destroy_thread_queue_wp( void *const p );

/**
 * @class NCThreadQueueWpKey
 *
 * @brief
 */
class __attribute__( ( visibility( "default" ) ) ) NCThreadQueueWpKey {
   public:
    /**
     * Get the NCThreadQueueWpKey's instance. If it not exist, create it.
     *
     * @return the instance.
     */
    static NCThreadQueueWpKey *instance();

    /**
     * Destroy the instance.
     */
    static VOID destroy();

    /**
     * Get the thread object address where the thread key created.
     *
     * @return  the private data.
     */
    VOID *getQueueWp() const;

    /**
     * Associate with thread's private data and the thread's key.
     *
     * @param  thread's private data
     */
    VOID setQueueWp( const VOID *const pThread ) const;

   private:
    NCThreadQueueWpKey();
    virtual ~NCThreadQueueWpKey();
    pthread_key_t              m_key;
    static NCThreadQueueWpKey *s_pInstance;
    static NCSyncObj           s_cSync;
};
OSAL_END_NAMESPACE

#endif  // INCLUDE_NCORE_NCTHREADQUEUEWPKEY_H_
/* EOF */
