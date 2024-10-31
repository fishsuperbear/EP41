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
 * @file NCErrorShmOperator.h
 * @brief
 * @date 2020-05-09
 *
 */

#ifndef NCERRORSHMOPERATOR_H
#define NCERRORSHMOPERATOR_H

#ifndef NCERRORPUBDEF_H
#include "osal/ncore/NCErrorPubDef.h"
#endif

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE

class NCErrorShmCfg;

/**
 * @brief
 *
 * @class NCErrorShmOperator
 */
class __attribute__( ( visibility( "default" ) ) ) NCErrorShmOperator {
   public:
    /**
     * @brief Construct a new NCErrorShmOperator object
     *
     * @param errType[IN]:error type
     * @param debugState[IN]:open or close debug log
     * @param createFlg [IN]:
     *    NC_TRUE: create a new share memory if share memory isn't exist.
     *    NC_FALSE: not create a new share memory if share memory isn't exist
     */
    NCErrorShmOperator( NC_ERRORTYPE errType, NC_BOOL debugState = NC_FALSE,
                        NC_BOOL createFlg = NC_FALSE );

    /**
     * @brief Destroy the NCErrorShmOperator object
     *
     */
    virtual ~NCErrorShmOperator();

    /**
     * @brief set debug log state
     *
     * @param    state[IN]:open or close
     * @return   VOID
     */
    VOID setDebugState( NC_BOOL state );

    /**
     * @brief get debug log state
     *
     * @param    VOID
     * @return   NC_BOOL:open or close
     */
    NC_BOOL getDebugState();

    /**
     * @brief get log number
     *
     * @param    VOID
     * @return   log number
     */
    INT32 length();

    /**
     * @brief get max log number
     *
     * @param    VOID
     * @return   max log number
     */
    INT32 getMaxLength();

    /**
     * @brief get log buffer size
     *
     * @param    VOID
     * @return   log buffer size
     */
    INT32 size();

    /**
     * @brief insert a log
     *
     * @param    data[IN]:log data
     * @return   VOID
     */
    VOID insert( const NCErrorLogFormat *data );

    /**
     * @brief get number of  NCErrorShmHeader
     *
     * @param    VOID
     * @return   number of NCErrorShmHeader
     */
    INT32 getNumber();

    /**
     * @brief copy all data to buf
     *
     * @param    buf[IN]:buffer
     * @param    dataSize[IN]:buffer data
     * @return   the number of copyed message
     */
    INT32 cpyAllData( VOID *buf, INT32 dataSize );

   private:
    /**
     * create mem
     *
     * @param    state[IN]:open or close
     * @return    void
     */
    VOID create( NC_BOOL debugState );

   private:
    // copy constructor
    NCErrorShmOperator( const NCErrorShmOperator & );
    // operator =
    NCErrorShmOperator &operator=( const NCErrorShmOperator & );

   private:
    // share memory address
    VOID *m_shmAddr;
    // error type
    NC_ERRORTYPE m_errType;
    // create state
    NC_BOOL m_create;
    // config
    const NCErrorShmCfg *m_shmCfg;
    // mem size
    INT32 m_totalSize;
    // fd
    INT32 m_fd;
};
OSAL_END_NAMESPACE

#endif /* NCERRORSHMOPERATOR_H */
       /* EOF */