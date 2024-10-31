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
 * @file dlt_protocol.h
 * @brief dlt_protocol
 * @date 2020-05-20
 *
 */

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif
#ifndef DLT_PROTOCOL_H
#define DLT_PROTOCOL_H
#include <stdint.h>

#include "osal/dlt/dlt_conf.h"

namespace dlt {
#if !defined( PACKED )
#define PACKED __attribute__( ( aligned( 1 ), packed ) )
#endif

enum StringType { ASCII_STRING = 0, UTF8_STRING = 1 };

const uint16_t DLT_LOG_LEN_MAX = 0XFFFFU;

typedef enum {
    DLT_FORMAT_DEFAULT    = 0x00, /**< no sepecial format */
    DLT_FORMAT_HEX8       = 0x01, /**< Hex 8 */
    DLT_FORMAT_HEX16      = 0x02, /**< Hex 16 */
    DLT_FORMAT_HEX32      = 0x03, /**< Hex 32 */
    DLT_FORMAT_HEX64      = 0x04, /**< Hex 64 */
    DLT_FORMAT_BIN8       = 0x05, /**< Binary 8 */
    DLT_FORMAT_BIN16      = 0x06, /**< Binary 16  */
    DLT_FORMAT_BIN32      = 0x07, /**< Binary 32  */
    DLT_FORMAT_BIN64      = 0x08, /**< Binary 64  */
    DLT_FORMAT_INT_DEC8   = 0x09, /**< Decimal int8 */
    DLT_FORMAT_INT_DEC16  = 0x0A, /**< Decimal int16  */
    DLT_FORMAT_INT_DEC32  = 0x0B, /**< Decimal int32  */
    DLT_FORMAT_INT_DEC64  = 0x0C, /**< Decimal int64  */
    DLT_FORMAT_UINT_DEC8  = 0x0D, /**< Decimal uint8 */
    DLT_FORMAT_UINT_DEC16 = 0x0E, /**< Decimal uint16  */
    DLT_FORMAT_UINT_DEC32 = 0x0F, /**< Decimal uint32  */
    DLT_FORMAT_UINT_DEC64 = 0x10, /**< Decimal uint64  */
    DLT_FORMAT_MAX                /**< maximum value, used for range check */
} DltFormatType;

#define DLT_HTYP_MSBF 0x02 /**< MSB first */

#define DLT_SWAP_16( value ) ( ( ( ( value ) >> 8U ) & 0xffU ) | ( ( ( value ) << 8U ) & 0xff00U ) )
#define DLT_SWAP_32( value )                                                   \
    ( ( ( ( value ) >> 24U ) & 0xffU ) | ( ( ( value ) << 8U ) & 0xff0000U ) | \
      ( ( ( value ) >> 8U ) & 0xff00U ) | ( ( ( value ) << 24U ) & 0xff000000U ) )

#define DLT_SWAP_64( value )                                           \
    ( ( ( (uint64) DLT_SWAP_32( (value) &0xffffffffullU ) ) << 32U ) | \
      ( DLT_SWAP_32( ( value ) >> 32U ) ) )

#define DLT_HTOBE_64( x ) DLT_SWAP_64( ( x ) )
#define DLT_HTOBE_32( x ) DLT_SWAP_32( ( x ) )
#define DLT_HTOBE_16( x ) DLT_SWAP_16( ( x ) )

typedef enum : int8_t {
    DLT_LOG_DEFAULT = -1,   /**< Default log level */
    DLT_LOG_OFF     = 0x00, /**< Log level off */
    DLT_LOG_FATAL   = 0x01, /**< fatal system error */
    DLT_LOG_ERROR   = 0x02, /**< error with impact to correct functionality */
    DLT_LOG_WARN    = 0x03, /**< warning, correct behaviour could not be ensured */
    DLT_LOG_INFO    = 0x04, /**< informational */
    DLT_LOG_DEBUG   = 0x05, /**< debug  */
    DLT_LOG_VERBOSE = 0x06, /**< highest grade of information */
    DLT_LOG_MAX             /**< maximum value, used for range check */
} DltLogLevelType;

const uint32_t DLT_ID_SIZE = 4U;
#define DLT_SIZE_WEID DLT_ID_SIZE
#define DLT_SIZE_WSID ( sizeof( UINT32 ) )
#define DLT_SIZE_WTMS ( sizeof( UINT32 ) )

#define DLT_MSIN_MSTP_SHIFT 1 /**< shift right offset to get mstp value */
#define DLT_MSIN_MTIN_SHIFT 4 /**< shift right offset to get mtin value */

#define DLT_HTYP_DEF 0X00
#define DLT_MSIN_VERB 0x01 /**< verbose */

/*
 * Definitions of mstp parameter in extended header.
 */
#define DLT_TYPE_LOG 0x00       /**< Log message type */
#define DLT_TYPE_APP_TRACE 0x01 /**< Application trace message type */
#define DLT_TYPE_NW_TRACE 0x02  /**< Network trace message type */
#define DLT_TYPE_CONTROL 0x03   /**< Control message type */

#define DLT_CONTROL_REQUEST 0x01  /**< Request message */
#define DLT_CONTROL_RESPONSE 0x02 /**< Response to request message */

/*
 * Definitions of types of arguments in payload.
 */
#define DLT_TYPE_INFO_TYLE                                                      \
    0x0000000f /**< Length of standard data: 1 = 8bit, 2 = 16bit, 3 = 32 bit, 4 \
                  = 64 bit, 5 = 128 bit */
#define DLT_TYPE_INFO_BOOL 0x00000010U /**< Boolean data */
#define DLT_TYPE_INFO_SINT 0x00000020U /**< Signed integer data */
#define DLT_TYPE_INFO_UINT 0x00000040U /**< Unsigned integer data */
#define DLT_TYPE_INFO_FLOA 0x00000080U /**< Float data */
#define DLT_TYPE_INFO_ARAY 0x00000100U /**< Array of standard types */
#define DLT_TYPE_INFO_STRG 0x00000200U /**< String */
#define DLT_TYPE_INFO_RAWD 0x00000400U /**< Raw data */
#define DLT_TYPE_INFO_VARI \
    0x00000800 /**< Set, if additional information to a variable is available */
#define DLT_TYPE_INFO_FIXP 0x00001000U /**< Set, if quantization and offset are added */
#define DLT_TYPE_INFO_TRAI 0x00002000U /**< Set, if additional trace information is added */
#define DLT_TYPE_INFO_STRU 0x00004000U /**< Struct */
#define DLT_TYPE_INFO_SCOD 0x00038000U /**< coding of the type string: 0 = ASCII, 1 = UTF-8 */

#define DLT_TYLE_8BIT 0x00000001U
#define DLT_TYLE_16BIT 0x00000002U
#define DLT_TYLE_32BIT 0x00000003U
#define DLT_TYLE_64BIT 0x00000004U
#define DLT_TYLE_128BIT 0x00000005U

#define DLT_SCOD_ASCII 0x00000000U
#define DLT_SCOD_UTF8 0x00008000U
#define DLT_SCOD_HEX 0x00010000U
#define DLT_SCOD_BIN 0x00018000U

/*
 * Definitions of DLT services.
 */
#define DLT_SERVICE_ID_SET_LOG_LEVEL 0x01U         /**< Service ID: Set log level */
#define DLT_SERVICE_ID_SET_TRACE_STATUS 0x02U      /**< Service ID: Set trace status */
#define DLT_SERVICE_ID_GET_LOG_INFO 0x03U          /**< Service ID: Get log info */
#define DLT_SERVICE_ID_GET_DEFAULT_LOG_LEVEL 0x04U /**< Service ID: Get dafault log level */
#define DLT_SERVICE_ID_STORE_CONFIG 0x05U          /**< Service ID: Store configuration */
#define DLT_SERVICE_ID_RESET_TO_FACTORY_DEFAULT      \
    0x06U /**< Service ID: Reset to factory defaults \
             */
#define DLT_SERVICE_ID_SET_COM_INTERFACE_STATUS \
    0x07U /**< Service ID: Set communication interface status */
#define DLT_SERVICE_ID_SET_COM_INTERFACE_MAX_BANDWIDTH \
    0x08U /**< Service ID: Set communication interface maximum bandwidth */
#define DLT_SERVICE_ID_SET_VERBOSE_MODE 0x09U         /**< Service ID: Set verbose mode */
#define DLT_SERVICE_ID_SET_MESSAGE_FILTERING 0x0AU    /**< Service ID: Set message filtering */
#define DLT_SERVICE_ID_SET_TIMING_PACKETS 0x0BU       /**< Service ID: Set timing packets */
#define DLT_SERVICE_ID_GET_LOCAL_TIME 0x0CU           /**< Service ID: Get local time */
#define DLT_SERVICE_ID_USE_ECU_ID 0x0DU               /**< Service ID: Use ECU id */
#define DLT_SERVICE_ID_USE_SESSION_ID 0x0EU           /**< Service ID: Use session id */
#define DLT_SERVICE_ID_USE_TIMESTAMP 0x0FU            /**< Service ID: Use timestamp */
#define DLT_SERVICE_ID_USE_EXTENDED_HEADER 0x10U      /**< Service ID: Use extended header */
#define DLT_SERVICE_ID_SET_DEFAULT_LOG_LEVEL 0x11U    /**< Service ID: Set default log level */
#define DLT_SERVICE_ID_SET_DEFAULT_TRACE_STATUS 0x12U /**< Service ID: Set default trace status */
#define DLT_SERVICE_ID_GET_SOFTWARE_VERSION 0x13U     /**< Service ID: Get software version */
#define DLT_SERVICE_ID_MESSAGE_BUFFER_OVERFLOW 0x14U  /**< Service ID: Message buffer overflow */
#define DLT_SERVICE_ID_UNREGISTER_CONTEXT 0x0f01U     /**< Service ID: Message unregister context */
#define DLT_SERVICE_ID_CONNECTION_INFO 0xf02U         /**< Service ID: Message connection info */
#define DLT_SERVICE_ID_TIMEZONE 0xf03U                /**< Service ID: Timezone */
#define DLT_SERVICE_ID_MARKER 0xf04U                  /**< Service ID: Timezone */
#define DLT_SERVICE_ID_CALLSW_CINJECTION 0xFFFU /**< Service ID: Message Injection (minimal ID) */

#define DLT_SERVICE_ID_USER_SETFILTER ( 0xFFF + 1 )     /**< Service ID: SetFilter */
#define DLT_SERVICE_ID_USER_DUMPSTATISTIC ( 0xFFF + 2 ) /**< Service ID: DumpStatistic */
#define DLT_SERVICE_ID_USER_HANDLECFGINFO ( 0xFFF + 3 ) /**< Service ID: SaveCfgInfo */
#define DLT_SERVICE_ID_USER_CHANGEMODE ( 0xFFF + 4 )    /**< Service ID: ChangeMode */
#define DLT_SERVICE_ID_USER_CLEARCACHE ( 0xFFF + 5 )    /**< Service ID: ClearCache */

/*
 * Definitions of DLT service response status
 */
#define DLT_SERVICE_RESPONSE_OK 0x00U            /**< Control message response: OK */
#define DLT_SERVICE_RESPONSE_NOT_SUPPORTED 0x01U /**< Control message response: Not supported */
#define DLT_SERVICE_RESPONSE_ERROR 0x02U         /**< Control message response: Error */
}  // namespace dlt
#endif
