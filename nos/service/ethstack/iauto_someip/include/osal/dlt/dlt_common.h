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
 * @file dlt_common.h
 * @brief
 * @date 2020-06-18
 *
 */
#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif
#ifndef DLT_COMMON_H
#define DLT_COMMON_H

#include <memory.h>

#include <string>

#include "osal/dlt/dlt_conf.h"
#include "osal/dlt/dlt_protocol.h"
#include <stdint.h>

namespace dlt {
/**
 * @class DLTStorageHeader
 *
 * @brief Define of DLTStorageHeader
 */
typedef struct {
    uint8_t  iden[ DLT_ID_SIZE ]; /**< This iden should be DLT0x01 */
    uint32_t seconds;             /**< seconds since 1.1.1970 */
    uint32_t microseconds;        /**< Microseconds */
    uint8_t  ecu[ DLT_ID_SIZE ];  /**< The ECU id is added, if it is not already in the
                                   DLT message itself */
} PACKED DLTStorageHeader;

/**
 * @class DLTStandardHeader
 *
 * @brief Define of DLTStandardHeader
 */
typedef struct {
    struct HTYP_T {
        uint8_t UEH : 1;
        uint8_t MSBF : 1;
        uint8_t WEID : 1;
        uint8_t WSID : 1;
        uint8_t WTMS : 1;
        uint8_t VERS : 3;
    } HTYP;
    uint8_t  MCNT;
    uint16_t LEN;
} PACKED DLTStandardHeader;

/**
 * @class DLTStandardHeader_ECU
 *
 * @brief Define of DLTStandardHeader_ECU
 */
typedef struct {
    uint8_t ECU[ DLT_ID_SIZE ];
} PACKED DLTStandardHeader_ECU;

/**
 * @class DLTStandardHeader_SEID
 *
 * @brief Define of DLTStandardHeader_SEID
 */
typedef struct {
    uint32_t SEID;
} PACKED DLTStandardHeader_SEID;

/**
 * @class DLTStandardHeader_TMSP
 *
 * @brief Define of DLTStandardHeader_TMSP
 */
typedef struct {
    uint32_t TMSP;
} PACKED DLTStandardHeader_TMSP;

/**
 * @class DLTExtendedHeader
 *
 * @brief Define of DLTExtendedHeader
 */
typedef struct {
    struct MSIN_T {
        uint8_t VERB : 1;  // MSIN:0    Verbose:DLT_MSIN_VERB
        uint8_t MSTP : 3;  // MSIN:1-3  Message Type:DLT_TYPE_XX
        uint8_t MTIN : 4;  // MSIN:4-7  Message Type Info: LogLevel(LogType)
    } MSIN;

    uint8_t NOAR;
    uint8_t APID[ DLT_ID_SIZE ];  // app id
    uint8_t CTID[ DLT_ID_SIZE ];  // context id
} PACKED DLTExtendedHeader;

typedef enum : uint8_t {
    DLT_CHANNAL_MAIN   = 0x00, /**< channal main */
    DLT_CHANNAL_SYSTEM = 0x01, /**< channal system */
    DLT_CHANNAL_EVENT  = 0x02, /**< channal event */
    DLT_CHANNAL_RADIO  = 0x03, /**< channal radio */
    DLT_CHANNAL_MAX            /**< maximum value, used for range check */
} DltLogChannal;

/**
 * @class DLTUserDefinedHeader
 *
 * @brief Define of DLTUserDefinedHeader
 */
const uint32_t DLT_USERDEF_TAG_SIZE = 16U;

typedef struct {
    uint8_t       iden[ DLT_ID_SIZE ];  // This iden should be 0x12 34 56 78 */
    DltLogChannal Channal;              // DltLogChannel
    char          Tag[ DLT_USERDEF_TAG_SIZE ];
    uint64_t      Pid;
    uint64_t      Tid;
    uint32_t      Seconds;      /**< seconds since 1.1.1970 */
    uint32_t      Microseconds; /**< Microseconds */
} PACKED DLTUserDefinedHeader;

typedef struct {
    std::string ECUId;
    std::string AppId;
} DltLogCfgInfo;

typedef struct {
    const DLTStandardHeader *     Standard;
    const DLTStandardHeader_ECU * StandardEcu;
    const DLTStandardHeader_SEID *StandardSeid;
    const DLTStandardHeader_TMSP *StandardTmsp;

    const DLTExtendedHeader *Extended;
#ifdef _HAVE_USERDEFINE_HEADER
    const DLTUserDefinedHeader *UserDefined;
#endif
    const uint8_t *Payload;
} DltLogItem;

inline uint16_t GetLogSize( const DltLogItem *item ) {
    uint16_t ret = 0U;
    if ( ( item != nullptr ) && ( item->Standard != nullptr ) ) {
        ret = DLT_HTOBE_16( item->Standard->LEN );
    }
    return ret;
}

inline uint8_t GetLogType( const DltLogItem *item ) {
    uint8_t ret = 0xFFU;
    if ( ( item != nullptr ) && ( item->Extended != nullptr ) ) {
        ret = item->Extended->MSIN.MSTP;
    }
    return ret;
}

inline uint8_t GetCtrlType( const DltLogItem *item ) {
    uint8_t ret = 0xFFU;
    if ( ( item != nullptr ) && ( item->Extended != nullptr ) ) {
        ret = item->Extended->MSIN.MTIN;
    }
    return ret;
}

inline uint32_t GetServiceID( const DltLogItem *item ) {
    uint32_t ret = 0;

    if ( ( item != nullptr ) && ( item->Payload != nullptr ) ) {
        if ( ( item->Extended != nullptr ) && ( item->Extended->MSIN.VERB == 1 ) ) {
            // verbose mode
            ret = *( reinterpret_cast<const uint32_t *>( item->Payload ) + sizeof( uint32_t ) );
        } else {
            // non-verbose mode
            ret = *( reinterpret_cast<const uint32_t *>( item->Payload ) );
        }
        if ( 1U == item->Standard->HTYP.MSBF ) {
            ret = DLT_HTOBE_32( ret );
        }
    }
    return ret;
}

inline uint16_t GetPayLoadSize( const DltLogItem *item ) {
    uint16_t ret = 0;
    ret          = GetLogSize( item );
    if ( item != nullptr ) {
        ret -= sizeof( DLTStandardHeader );
        if ( item->StandardEcu != nullptr ) {
            ret -= sizeof( DLTStandardHeader_ECU );
        }
        if ( item->StandardSeid != nullptr ) {
            ret -= sizeof( DLTStandardHeader_SEID );
        }
        if ( item->StandardTmsp != nullptr ) {
            ret -= sizeof( DLTStandardHeader_TMSP );
        }
        if ( item->Extended != nullptr ) {
            ret -= sizeof( DLTExtendedHeader );
        }
#ifdef _HAVE_USERDEFINE_HEADER
        if ( item->UserDefined != nullptr ) {
            ret -= sizeof( DLTUserDefinedHeader );
        }
#endif
    }
    return ret;
}
}  // namespace dlt
#endif