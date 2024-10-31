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
 * @file NCRunnableLooper.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCRUNNABLELOOPER_H_
#define INCLUDE_NCORE_NCRUNNABLELOOPER_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include "osal/ncore/NCGlobalAPI.h"
#include "osal/ncore/NCRunnable.h"
#include "osal/ncore/NCRunnableQueue.h"
#include "osal/ncore/NCTypesDefine.h"

OSAL_BEGIN_NAMESPACE
/**
 * @class NCRunnableLooper
 *
 * @brief Class for NCRunnableLooper
 * **/
class __attribute__( ( visibility( "default" ) ) ) NCRunnableLooper {
   public:
    /**
     * @brief Construct a new NCRunnableLooper
     *
     * @param runnableQueue runnable Queue(weak pointer)
     */
    explicit NCRunnableLooper( NCRunnableQueueWeakHolder runnableQueue );

    /**
     * @brief Construct a new NCRunnableLooper
     *
     * @param rhs other NCRunnableLooper
     */
    NCRunnableLooper( const NCRunnableLooper &rhs );

    /**
     * @brief Destroy the NCRunnableLooper object
     */
    virtual ~NCRunnableLooper();

    /**
     * @brief post runnable at tail of queue
     *
     * @param runnable the runnable data want to post
     * @return NC_BOOL True:success False:otherwise
     */
    virtual NC_BOOL postRunnable( NCRunnableHolder runnable );

    /**
     * @brief post runnable at header of queue
     *
     * @param runnable the runnable data want to post
     * @return NC_BOOL True:success False:otherwise
     */
    virtual NC_BOOL postRunnableAtFront( NCRunnableHolder runnable );

    /**
     * @brief remove all same runnables in both queue and delayed queue
     *
     * @param runnable the runnable want to remove
     * @return NC_BOOL True:success False:otherwise
     */
    virtual NC_BOOL removeRunnable( NCRunnableHolder runnable );

    /**
     * @brief post a runnable to run after 'delayed' ms
     *
     * @param runnable the runnable want to remove
     * @param delayed the delayed timer
     * @return NC_BOOL True:success False:otherwise
     */
    virtual NC_BOOL postRunnableDelayed( NCRunnableHolder runnable, UINT32 delayed );

    /**
     * @brief check looper is valid or not
     *
     * @return NC_BOOL True:valid False:otherwise
     */
    NC_BOOL isValid() const;

    /**
     * @brief assignment function create a new NCRunnableLooper from the "rhs"
     *
     * @param rhs the other NCRunnableLooper object
     * @return NCRunnableLooper& this NCRunnableLooper
     */
    NCRunnableLooper &operator=( const NCRunnableLooper &rhs );

   private:
    NCRunnableQueueWeakHolder m_runnableQueue;
    NCRunnableLooper();
};
OSAL_END_NAMESPACE
#endif  // INCLUDE_NCORE_NCRUNNABLELOOPER_H_
/* EOF */
