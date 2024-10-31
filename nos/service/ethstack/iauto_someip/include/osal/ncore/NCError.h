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
 * @file NCError.h
 * @brief
 * @date 2020-05-09
 *
 */

#ifndef NCERROR_H
#define NCERROR_H

#include "osal/ncore/NCErrorOperator.h"
#include "osal/ncore/NCErrorPubDef.h"
#include "osal/ncore/NCErrorSetting.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE
/**
 * @brief
 *
 * @class NCError
 */
class __attribute__( ( visibility( "default" ) ) ) NCError {
   public:
    /**
     * record recoverable error log
     *
     * @param file [IN]:file information
     * @param line [IN]line No
     * @param error [IN]Error No
     * @param option [IN]option
     * @return VOID
     */
    static VOID recordError( const CHAR* file, INT32 line, NC_ERROR error, INT32 option );

    /**
     * record Debug error log
     *
     * @param file [IN]:file information
     * @param line [IN]line No
     * @param error [IN]Error No
     * @param option [IN]option
     * @return VOID
     */
    static VOID recordDebug( const CHAR* file, INT32 line, NC_ERROR error, INT32 option );

    /**
     * record Fatal error log
     *
     * @param file [IN]:file information
     * @param line [IN]line No
     * @param error [IN]Error No
     * @param option [IN]option
     * @return VOID
     */
    static VOID recordFatal( const CHAR* file, INT32 line, NC_ERROR error, INT32 option );

    /**
     * get the error num
     *
     * @param debug [IN]: debug level error number
     * @param error [IN]: error level error number
     * @param fatal [IN]: fatal level error number
     */
    static VOID getErrorNums( INT32* debug, INT32* error, INT32* fatal );

    /**
     * close fd of errd
     *
     * @param  VOID
     * @return TRUE indicate success, vis versa.
     */
    static NC_BOOL closeErrdFd();

   private:
    // constructor
    NCError();
    // destructor
    ~NCError();
    // copy constructor
    NCError( const NCError& );
    // operator =
    NCError& operator=( const NCError& );

   private:
    // error operation objects
    static NCErrorOperator s_errOperator;
    static NCErrorOperator s_debugOperator;
    static NCErrorOperator s_fatalOperator;
};

/**
 * @brief Record recoverable errors macro
 *
 * It can automatically append the file info and line number.
 * <b>You should use this macro instead using AL_AplError::recordError function directly.</b>
 */
#define NCErrorLog( error, option ) NCError::recordError( __FILE__, __LINE__, error, option )

/**
 * @brief Record Debug errors macro
 *
 * It can automatically append the file info and line number.
 * <b>You should use this macro instead using AL_AplError::recordError function directly.</b>
 */
#define NCDebugError( error, option ) NCError::recordDebug( __FILE__, __LINE__, error, option )

/**
 * @brief Record Fatal errors macro
 *
 * It can automatically append the file info and line number.
 * <b>You should use this macro instead using AL_AplError::recordError function directly.</b>
 */
#define NCFatalError( error, option ) NCError::recordFatal( __FILE__, __LINE__, error, option )
OSAL_END_NAMESPACE

#endif /* NCERROR_H */
       /* EOF */