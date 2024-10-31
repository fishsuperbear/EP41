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
 * @file NCAutoSync.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCAUTOSYNC_H_
#define INCLUDE_NCORE_NCAUTOSYNC_H_

#include <pthread.h>

#include "osal/ncore/NCSyncObj.h"
#include "osal/ncore/NCTypesDefine.h"
#include "osal/ncore/NCWaitObj.h"
#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE
// Class declaration
class NCAutoSync;

/**
 * @brief
 *
 * @class NCAutoSync
 *
 * The class to synchronize automatically.
 *
 * @code
 *
 * Please use the class of NCAutoSync like below:
 *
 * VOID
 * XXXXClass::XXXfunction()
 * {
 *     ...
 *     NCAutoSync cSync(s_cSync); // s_cSync is the object of NCSyncObj
 *     ...
 * }
 * @endcode
 *
 */
class __attribute__( ( visibility( "default" ) ) ) NCAutoSync {
   public:
    /**
     * Construction.
     *
     * @param cSync : The object of class NCSyncObj.
     */
    explicit NCAutoSync( NCSyncObj &cSync );

    /**
     * Destruction.
     */
    ~NCAutoSync();

   private:
    NCSyncObj &m_cSync;

    /**
     * @brief Construct a new NCAutoSync object
     *
     * @param cSync other NCAutoSync
     */
    NCAutoSync( const NCAutoSync &cSync );

    /**
     * @brief Construct a new NCAutoSync object
     *
     * @param cSync other NCAutoSync
     * @return NCAutoSync&
     */
    NCAutoSync &operator=( const NCAutoSync &cSync );
};

struct NCOnceFlag {
    NCOnceFlag() : mOnceFlag( PTHREAD_ONCE_INIT ){};

    NCOnceFlag( const NCOnceFlag & ) = delete;
    NCOnceFlag &operator=( const NCOnceFlag & ) = delete;

    pthread_once_t mOnceFlag;
};

void NCCallOnce( NCOnceFlag &OnceFlag, VOID ( *INIT_ROUTINE )( VOID ) );

OSAL_END_NAMESPACE
#endif  // INCLUDE_NCORE_NCAUTOSYNC_H_
/* EOF */
