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
 * @file NCRunnableQueue.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCRUNNABLEQUEUE_H_
#define INCLUDE_NCORE_NCRUNNABLEQUEUE_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include "osal/ncore/NCGlobalAPI.h"
#include "osal/ncore/NCRefBase.h"
#include "osal/ncore/NCRunnable.h"
#include "osal/ncore/NCTypesDefine.h"

OSAL_BEGIN_NAMESPACE
class NCRunnableQueueImpl;
class NCRunnableQueueOwner {
   public:
    /**
     * @brief Construct a new NCRunnableQueueOwner
     */
    NCRunnableQueueOwner();

    /**
     * @brief Destroy the NCRunnableQueueOwner
     */
    virtual ~NCRunnableQueueOwner();

    /**
     * @brief notify
     *
     * @return VOID
     */
    virtual VOID notify() = 0;

    /**
     * @brief notify warning
     *
     * @param level warning level
     * @return VOID
     */
    virtual VOID notifyWarning( UINT32 level );
};

class __attribute__( ( visibility( "default" ) ) ) NCRunnableQueue {  // : public NCRefBase
   public:
    /**
     * @brief Construct a new NCRunnableQueue
     *
     * @param ower  NCRunnableQueueOwner pointer
     */
    explicit NCRunnableQueue( NCRunnableQueueOwner *const ower );

    /**
     * @brief Destroy the NCRunnableQueue
     */
    virtual ~NCRunnableQueue();

   public:
    /**
     * @brief add runnable to the front of list
     *
     * @param runnable  the NCRunnableHolder want to add
     * @return NC_BOOL NC_TRUE:sucess NC_FALSE:otherwise
     */
    NC_BOOL enqueueAtFront( NCRunnableHolder runnable );

    /**
     * @brief add runnable to the tail of list
     *
     * @param runnable  the NCRunnableHolder want to add
     * @return NC_BOOL NC_TRUE:sucess NC_FALSE:otherwise
     */
    NC_BOOL enqueueAtTail( NCRunnableHolder runnable );

    /**
     * @brief remove the runnable from the list
     *
     * @param runnable the NCRunnableHolder want to remove
     * @return NC_BOOL NC_BOOL NC_TRUE:sucess NC_FALSE:otherwise
     */
    NC_BOOL removeRunnable( NCRunnableHolder runnable );

    /**
     * @brief clear all element
     *
     * @return VOID
     */
    VOID clearAll();

    /**
     * @brief get first element from list
     *
     * @return NCRunnableHolder get the element
     */
    NCRunnableHolder deQueue();

    /**
     * @brief delay add runnable to the list
     *
     * @param runnable  the NCRunnableHolder want to add
     * @param timeout   the delay timer
     * @return NC_BOOL  NC_BOOL NC_TRUE:sucess NC_FALSE:otherwise
     */
    NC_BOOL enqueueDelayed( NCRunnableHolder runnable, UINT64 timeout );

    /**
     * @brief returned timeout to wait
     *
     * @param runnable [OUT] geth the runnable
     * @return UINT64   the timer should to wait
     */
    UINT64 deQueue( NCRunnableHolder &runnable );

   private:
    NCRunnableQueueImpl *m_impl;
    NCRunnableQueue();
};

typedef ncsp<NCRunnableQueue>::sp NCRunnableQueueStrongHolder;
typedef ncsp<NCRunnableQueue>::wp NCRunnableQueueWeakHolder;
OSAL_END_NAMESPACE
#endif  // INCLUDE_NCORE_NCRUNNABLEQUEUE_H_
/* EOF */
