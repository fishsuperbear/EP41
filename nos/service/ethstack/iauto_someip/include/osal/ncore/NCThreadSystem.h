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
 * @file NCThreadSystem.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCTHREADSYSTEM_H_
#define INCLUDE_NCORE_NCTHREADSYSTEM_H_

#include "osal/ncore/NCList.h"
#include "osal/ncore/NCThreadSystemIF.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE

/**
 * @brief
 *
 * @class NCThreadSystem
 */
class __attribute__( ( visibility( "default" ) ) ) NCThreadSystem : public NCThreadSystemIF {
   public:
    /**
     * Constructor
     */
    NCThreadSystem();

    /**
     * Destructor
     */
    virtual ~NCThreadSystem();

    /**
     * Register the thread to thread manager system.
     *
     * @param   name: The thread's name.
     * @param   handle:
     * @param   thread_id: thread ID.
     * @return  thread no.
     */
    virtual INT32 registerThread( const CHAR *const name, const VOID *const handle,
                                  UINT32 thread_id ) override;

    /**
     * Unregister the thread from thread manager system.
     *
     * @param   thread_no: The thread's num which come from the function
     *                     of registerThread.
     */
    virtual VOID unregisterThread( INT32 thread_no ) override;

    /**
     * Add thread table to thread manager system.
     *
     * @param   table: the table that contain
     *                 thread's name, priority, sanity check.
     * @return  NC_TRUE indicate success, vis versa.
     */
    virtual NC_BOOL addThreadTable( const NC_THREAD_TABLE *const table ) override;

    /**
     * Get thread infomation from thread table.
     *
     * @param   name: thread name.
     * @param   priority: thread priority.
     * @param   sanity: it mean if the thread is sanity.
     * @return  NC_TRUE indicate success, vis versa.
     */
    virtual NC_BOOL getThreadTableInfo( const CHAR *const name, UINT32 *const priority,
                                        UINT32 &sanity ) override;

   private:
    NCThreadSystem( const NCThreadSystem &src );
    NCThreadSystem &operator=( const NCThreadSystem &src );

    NC_BOOL compareName( const CHAR *const name, const CHAR *const matchname ) const;

    NCList<NC_THREAD_TABLE> m_threadTable;
};
OSAL_END_NAMESPACE
#endif  // INCLUDE_NCORE_NCTHREADSYSTEM_H_
/* EOF */
