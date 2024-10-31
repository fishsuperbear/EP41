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
 * @file NCRunnableThread.h
 * @brief
 * @date 2020-06-03
 *
 */

#ifndef NCRUNNABLETHREAD_H
#define NCRUNNABLETHREAD_H

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include <osal/ncore/NCGlobalAPI.h>

#include "osal/ncore/NCRunnableLooper.h"
#include "osal/ncore/NCRunnableQueue.h"
#include "osal/ncore/NCSyncObj.h"
#include "osal/ncore/NCThread.h"
#include "osal/ncore/NCTypesDefine.h"

OSAL_BEGIN_NAMESPACE

/**
 * @class NCRunnableThread
 *
 * @brief Define Runnable Thread..
 * **/
class __attribute__( ( visibility( "default" ) ) ) NCRunnableThread : public NCThread,
                                                                      NCRunnableQueueOwner {
   public:
    /**
     * @brief Get the Current Looper object
     *
     * @return NCRunnableLooper the current NCRunnableLooper
     */
    static NCRunnableLooper getCurrentLooper();

    /**
     * @brief Construct a new NCRunnableThread
     */
    NCRunnableThread();

    /**
     * @brief Destroy the NCRunnableThread
     */
    virtual ~NCRunnableThread();

    /**
     * @brief get the looper
     *
     * @return NCRunnableLooper
     */
    NCRunnableLooper looper();

    /**
     * @brief notify
     *
     * @return VOID
     */
    VOID notify();

    /**
     * @brief notify warning msg
     *
     * @param level warning level
     * @return VOID
     */
    VOID notifyWarning( uint level );

   protected:
    virtual void run();

   private:
    NCRunnableQueueStrongHolder m_runnableQueue;
    // for log
    UINT32    m_WarningLevel;
    NCSyncObj m_WarningSync;
};
OSAL_END_NAMESPACE
#endif
/*eof*/
