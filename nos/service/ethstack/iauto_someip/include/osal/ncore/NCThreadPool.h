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
 * @file NCThreadPool.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCTHREADPOOL_H_
#define INCLUDE_NCORE_NCTHREADPOOL_H_

#include "osal/ncore/NCTypesDefine.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE

class NCThreadPoolAttrs;
class NCThreadPoolImpl;

/**
 * @brief
 *
 * @class NCThreadPool
 *
 * Warning:
 * Only for QNX.
 * Forbidden to use in Linux : Empty implementation.
 *
 */
class NCThreadPool {
   public:
    explicit NCThreadPool( NCThreadPoolAttrs *const thrPoolAttrs );
    virtual ~NCThreadPool();

    VOID startThreadPool();
    VOID stopThreadPool();

    // priority : may set ? need test...
    INT32   getPriority() const;
    NC_BOOL setPriority( INT32 pri );

    NC_BOOL setLowThreadNums( INT32 lowNums );
    NC_BOOL setHighThreadNums( INT32 highNums );
    NC_BOOL setMaxThreadNums( INT32 maxThreadNums );
    NC_BOOL setIncThreadNums( INT32 incThreadNums );

    INT32 getLowThreadNums() const;
    INT32 getHighThreadNums() const;
    INT32 getMaxThreadNums() const;
    INT32 getIncThreadNums() const;

    NC_BOOL isAlive() const;

   private:
    NCThreadPoolImpl *m_threadPoolImpl;

    NCThreadPool( const NCThreadPool & );
    NCThreadPool &operator=( const NCThreadPool & );
};
OSAL_END_NAMESPACE

#endif  // INCLUDE_NCORE_NCTHREADPOOL_H_

/* EOF */
