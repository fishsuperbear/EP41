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
 * @file NCThreadPoolAttrs.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCTHREADPOOLATTRS_H_
#define INCLUDE_NCORE_NCTHREADPOOLATTRS_H_

#if ( defined __QNX__ ) || ( defined __QNXNTO__ )
#include <sys/dispatch.h>
#endif
#include "osal/ncore/NCTypesDefine.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE
#if ( defined __QNX__ ) || ( defined __QNXNTO__ )
static const UINT32 NC_THREADPOOL_NAME_LEN = 100U;
#elif ( defined linux ) || ( defined __linux__ )
static const UINT32 NC_THREADPOOL_NAME_LEN = 16U;
#endif

// using NCThreadPool_ALLOC_FUNC =
// THREAD_POOL_PARAM_T *(*context_alloc)(THREAD_POOL_HANDLE_T *handle);
// using NCThreadPool_BLOCK_FUNC =
// THREAD_POOL_PARAM_T *(*block_func) (THREAD_POOL_PARAM_T *ctp);
// using NCThreadPool_HANDLE_FUNC =
// INT (*handler_func)(THREAD_POOL_PARAM_T *ctp);
// using NCThreadPool_UNBLOCK_FUNC =
// VOID (*unblock_func)(THREAD_POOL_PARAM_T *ctp);
// using NCThreadPool_FREE_FUNC =
// VOID (*context_free) (THREAD_POOL_PARAM_T *ctp);

// typedef THREAD_POOL_PARAM_T
// *(*NCThreadPool_ALLOC_FUNC)(THREAD_POOL_HANDLE_T *handle);
// typedef THREAD_POOL_PARAM_T
// *(*NCThreadPool_BLOCK_FUNC) (THREAD_POOL_PARAM_T *ctp);
// typedef INT (*NCThreadPool_HANDLE_FUNC)(THREAD_POOL_PARAM_T *ctp);
// typedef VOID (*NCThreadPool_UNBLOCK_FUNC)(THREAD_POOL_PARAM_T *ctp);
// typedef VOID (*NCThreadPool_FREE_FUNC) (THREAD_POOL_PARAM_T *ctp);
#if ( defined linux ) || ( defined __linux__ )
typedef VOID THREAD_POOL_PARAM_T;
typedef VOID THREAD_POOL_HANDLE_T;
#endif
using NCThreadPool_ALLOC_FUNC   = THREAD_POOL_PARAM_T *(*) ( THREAD_POOL_HANDLE_T *handle );
using NCThreadPool_BLOCK_FUNC   = THREAD_POOL_PARAM_T *(*) ( THREAD_POOL_PARAM_T *ctp );
using NCThreadPool_HANDLE_FUNC  = INT32 ( * )( THREAD_POOL_PARAM_T *ctp );
using NCThreadPool_UNBLOCK_FUNC = VOID ( * )( THREAD_POOL_PARAM_T *ctp );
using NCThreadPool_FREE_FUNC    = VOID ( * )( THREAD_POOL_PARAM_T *ctp );

/**
 * @brief
 *
 * @class NCThreadPoolAttrs
 */
class NCThreadPoolAttrs {
   public:
    NCThreadPoolAttrs();
    virtual ~NCThreadPoolAttrs();

    /**
     * @brief
     * ** all setxxxNeeded must be call for setting handle or callback.
     *
     * a. the handle set by setHandleNeeded will be passed to
     * context_alloc.
     * b. context_alloc: called when a new thread is created.
     *      returns a pointer, which is then passed to
     *      the blocking function, block_func.
     * c. block_func: called when the thread is ready to block.
     *      returns the same type of pointer (returns and input parameter)
     *      that's passed to handler_func.
     * d. handler_func: The function is passed the pointer
     *      returned by block_func.
     * e. unblock_func: called to unblock threads.
     * f. context_free: called when the worker thread exits,
     *      to free the context allocated with context_alloc.
     * context_alloc, block_func, handler_func, unblock_func, context_free:
     *      all of them are forbidden to set NULL, if NULL, then crash.
     *
     */
    VOID setHandleNeeded( THREAD_POOL_HANDLE_T *const handle );
    VOID setAllocCallbackNeeded( NCThreadPool_ALLOC_FUNC allocFunc );
    VOID setBlockCallbackNeeded( NCThreadPool_BLOCK_FUNC blockFunc );
    VOID setHandleCallbackNeeded( NCThreadPool_HANDLE_FUNC handleFunc );
    VOID setUnblockCallbackNeeded( NCThreadPool_UNBLOCK_FUNC unblockFunc );
    VOID setFreeCallbackNeeded( NCThreadPool_FREE_FUNC freeFunc );

    /**
     * @brief
     * get xxx: get handdle or callback.
     *
     */
    THREAD_POOL_HANDLE_T *    getHandle() const;
    NCThreadPool_ALLOC_FUNC   getAllocCallback() const;
    NCThreadPool_BLOCK_FUNC   getBlockCallback() const;
    NCThreadPool_HANDLE_FUNC  getHandleCallback() const;
    NCThreadPool_UNBLOCK_FUNC getUnblockCallback() const;
    NCThreadPool_FREE_FUNC    getFreeCallback() const;

    /**
     * @brief
     * ** if not call bellow functions, it will use default value.
     * setPriority: the priority of threads in thread pool.
     * setLowThreadNums: The minimum number of threads
     *      that the pool should keep in the blocked state.
     * setHighThreadNums: The maximum number of threads
     *      to keep in a blocked state.
     * setMaxThreadNums: The maximum number of threads
     *      that the pool can create.
     * setIncThreadNums: The number of new threads created at one time.
     * setThreadPoolName: The name for the threads in the thread pool.
     *
     */
    VOID setPriority( INT32 pri );
    VOID setLowThreadNums( INT32 lowNums );
    VOID setHighThreadNums( INT32 highNums );
    VOID setMaxThreadNums( INT32 maxThreadNums );
    VOID setIncThreadNums( INT32 incThreadNums );
    VOID setThreadPoolName( const CHAR *const poolName );

    /**
     * @brief
     * get xxx: get data.
     *
     */
    INT32 getPriority() const;
    INT32 getLowThreadNums() const;
    INT32 getHighThreadNums() const;
    INT32 getMaxThreadNums() const;
    INT32 getIncThreadNums() const;
    VOID  getThreadPoolName( CHAR *const poolName ) const;

   private:
    INT32 m_pri;
    INT32 m_lowNums;
    INT32 m_highNums;
    INT32 m_maxThreadNums;
    INT32 m_incThreadNums;
    CHAR  m_poolName[ NC_THREADPOOL_NAME_LEN ];

    THREAD_POOL_HANDLE_T *    m_handle;
    NCThreadPool_ALLOC_FUNC   m_allocFunc;
    NCThreadPool_BLOCK_FUNC   m_blockFunc;
    NCThreadPool_HANDLE_FUNC  m_handleFunc;
    NCThreadPool_UNBLOCK_FUNC m_unblockFunc;
    NCThreadPool_FREE_FUNC    m_freeFunc;

    NCThreadPoolAttrs( const NCThreadPoolAttrs & );
    NCThreadPoolAttrs &operator=( const NCThreadPoolAttrs & );
};
OSAL_END_NAMESPACE

#endif  // INCLUDE_NCORE_NCTHREADPOOLATTRS_H_
/* EOF */
