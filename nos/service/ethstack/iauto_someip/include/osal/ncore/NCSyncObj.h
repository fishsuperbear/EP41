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
 * @file NCSyncObj.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCSYNCOBJ_H_
#define INCLUDE_NCORE_NCSYNCOBJ_H_

#include <pthread.h>

#include "osal/ncore/NCThreadSafety.h"
#include "osal/ncore/NCTypesDefine.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE
// Class declaration
class NCSyncObj;

/**
 * @brief
 *
 * @class NCSyncObj
 */
class __attribute__( ( visibility( "default" ) ) ) CAPABILITY( "mutex" ) NCSyncObj {
   public:
    /**
     * Construction.
     */
    NCSyncObj();

    /**
     * Destruction.
     */
    virtual ~NCSyncObj();

    /**
     * Synchronize start.
     */
    VOID syncStart() ACQUIRE();

    /**
     * Try synchronize start
     *
     * @return NP_BOOL : NC_TRUE means synchronize succeed, and NC_FALSE failed.
     */
    NC_BOOL trySyncStart() TRY_ACQUIRE( true );

    /**
     * Synchronize end.
     */
    VOID syncEnd() RELEASE();

   private:
    pthread_mutex_t     m_mutex;
    pthread_mutexattr_t m_attr;

   private:
    NCSyncObj( const NCSyncObj &src );
    NCSyncObj &operator=( const NCSyncObj &src );
};

OSAL_END_NAMESPACE
#endif  // INCLUDE_NCORE_NCSYNCOBJ_H_
/* EOF */
