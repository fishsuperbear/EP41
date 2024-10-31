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
 * @file NCErrorReader.h
 * @brief
 * @date 2020-05-09
 *
 */

#ifndef NCERRORREADER_H
#define NCERRORREADER_H

#ifndef NCERRORPUBDEF_H
#include "osal/ncore/NCErrorPubDef.h"
#endif

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE
class NCErrorShmOperator;

/**
 * @brief
 *
 * @class NCErrorReader
 */
class __attribute__( ( visibility( "default" ) ) ) NCErrorReader {
   public:
    /**
     * @brief Construct a new NCErrorReader object
     *
     * @param errType [IN]:error type
     */
    NCErrorReader( NC_ERRORTYPE errType );

    /**
     * @brief Destroy the NCErrorReader object
     *
     */
    virtual ~NCErrorReader();

    /**
     * @brief read all log into memory
     *
     * @param    VOID
     * @return    NC_BOOL True:success False:otherwise
     */
    NC_BOOL read();

    /**
     * @brief get log number
     *
     * @param   VOID
     * @return  the number of log
     */
    INT32 length();

    /**
     * @brief copy a log
     *
     * @param index [IN]:log index
     * @param *logInfo [IN/OUT]:log buffer
     * @return NC_BOOL
     */
    NC_BOOL getLog( INT32 index, NCErrorLogFormat *logInfo );

   private:
    // copy constructor
    NCErrorReader( const NCErrorReader & );
    // operator =
    NCErrorReader &operator=( const NCErrorReader & );

   private:
    // log type
    NC_ERRORTYPE m_errType;
    // share memory
    NCErrorShmOperator *m_shmOperator;
    // local buffer for saving log
    VOID *m_buf;
    // log number of local buffer
    INT32 m_length;
};
OSAL_END_NAMESPACE

#endif /* NCERRORREADER_H */
       /* EOF */