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
 * @file NCErrorSetting.h
 * @brief
 * @date 2020-05-09
 *
 */

#ifndef NCERRORSETTING_H
#define NCERRORSETTING_H

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include "osal/ncore/NCTypesDefine.h"

OSAL_BEGIN_NAMESPACE
/**
 * @brief
 *
 * @class NCErrorSetting
 */
class __attribute__( ( visibility( "default" ) ) ) NCErrorSetting {
   public:
    /**
     * @brief get debug log state
     *
     * @param    VOID
     * @return    NC_BOOL:open or close
     * @info no one calls this method to set debug log state, default debug log state will be used.
     */
    static NC_BOOL getDebugState();

    /**
     * @brief set debug log state
     *
     * @param    state[IN]:open or close
     * @return    void
     * @info no one calls this method to set debug log state, default debug log state will be used.
     */
    static VOID setDebugState( NC_BOOL state );
    /**
     * @brief close fd of errd
     *
     * @param VOID
     * @return TRUE indicate success, vis versa.
     */
    static NC_BOOL closeErrdFd();

   private:
    // constructor
    NCErrorSetting();
    // destructor
    ~NCErrorSetting();
    // copy constructor
    NCErrorSetting( const NCErrorSetting& );
    // operator =
    NCErrorSetting& operator=( const NCErrorSetting& );

   private:
    // write to file flag
    static NC_BOOL s_writeToFileFlag;
    // error level
    static NC_BOOL s_debugState;
    // has inited or not
    static NC_BOOL m_inited;
    // mutex
    static pthread_mutex_t m_nclogMutex;
};
OSAL_END_NAMESPACE
#endif /* NCERRORSETTING_H */
       /* EOF */