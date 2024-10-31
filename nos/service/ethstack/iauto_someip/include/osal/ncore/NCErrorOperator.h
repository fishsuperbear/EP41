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
 * @file NCErrorOperator.h
 * @brief
 * @date 2020-05-09
 *
 */

#ifndef NCERROROPERATOR_H
#define NCERROROPERATOR_H

#include <pthread.h>

#include "osal/ncore/NCErrorPubDef.h"
#include "osal/ncore/NCList.h"
#include "osal/ncore/NCSyncObj.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE
static const INT32 NC_ERROR_TAG_BUF_SIZE = 32;

static const INT32 NC_ERROR_CACHE_NUM = 10;

/**
 * @brief
 *
 * @class NCErrorOperator
 */
class __attribute__( ( visibility( "default" ) ) ) NCErrorOperator {
   public:
    /**
     * @brief Construct a new NCErrorOperator object
     *
     * @param errType  [IN]:error type
     */
    NCErrorOperator( NC_ERRORTYPE errType );

    /**
     * @brief Destroy the NCErrorOperator object
     *
     */
    virtual ~NCErrorOperator();

    /**
     * @brief record the error information
     *
     * @param *file [IN]:code file name
     * @param line  [IN]:code line
     * @param error [IN]:error id
     * @param option    [IN]:option
     * @return void
     *
     */
    VOID record( const CHAR* file, INT32 line, NC_ERROR error, INT32 option );

    /**
     * @brief Get the Error Number
     *
     * @param debug [OUT] the debug information
     * @param error [OUT] the error information
     * @param fatal [OUT] the fatal information
     * @return VOID
     */
    VOID getErrorNum( INT32* debug, INT32* error, INT32* fatal );

    /**
     * @brief send message
     *
     * @param msg   [IN] pointer of message
     * @param msgSize   [IN] the size of message
     * @param fromServer    [OUT] information from server
     * @return NC_BOOL True:send message success; False: otherwise
     */
    static NC_BOOL sendMsg( VOID* msg, INT32 msgSize, union ValuesFromServer* fromServer );

    /**
     * save msg to the queue
     * @param VOID* msg
     * @param INT msgSize
     * @return VOID
     *
     */

    /**
     * @brief add message to list
     *
     * @param msg   [IN] pointer of message
     * @param msgSize [IN] the size of message
     * @return VOID
     */
    VOID addMsg2List( const NCErrorLogInfoClient& msg, INT32 msgSize );

    /**
     * @brief send message from list
     *
     * @return VOID
     */
    VOID sendMsgFromList( VOID );

   private:
    /**
     * get time
     *
     * @param *buf[IN/OUT]:buffer
     * @param size[IN]:buffer size
     * @return void
     *
     */
    VOID formatTime( CHAR* buf, INT32 size );

   private:
    // copy constructor
    NCErrorOperator( const NCErrorOperator& );
    // operator =
    NCErrorOperator& operator=( const NCErrorOperator& );

   private:
    // error type
    NC_ERRORTYPE m_errType;
    // error tag
    CHAR                                   m_tag[ NC_ERROR_TAG_BUF_SIZE ];
    NCList<NCErrorLogInfoClient> m_msgList GUARDED_BY( m_mutex );
    NCSyncObj                              m_mutex;
};
OSAL_END_NAMESPACE

#endif /* NCERROROPERATOR_H */
       /* EOF */