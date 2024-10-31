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
 * @file dlt_payload.h
 * @brief
 * @date 2020-06-30
 *
 */
#ifndef DLT_PAYLOAD_H
#define DLT_PAYLOAD_H

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <memory>
#include <string>
#include "osal/dlt/dlt_common.h"
#include "osal/dlt/dlt_conf.h"
#include "osal/dlt/dlt_protocol.h"

namespace dlt {
enum PackMode { PackMode_NonVerbose = 0, PackMode_Verbose = 1 };

class DltPayload {
   public:
    DltPayload()          = default;
    virtual ~DltPayload() = default;

    virtual void SetMode( PackMode mode = PackMode_Verbose ) = 0;

    // for write
    template <typename T>
    uint32_t PackMsgPayload( const T &v, DltFormatType type = DLT_FORMAT_DEFAULT ) {
        return (packMsgPayloadInter)( reinterpret_cast<const uint8_t *>( &v ),
                                    static_cast<uint16_t>( sizeof( T ) ), (GetTypeInfo<T>)( type ) );
    };

    template <typename T = char *>
    uint32_t PackMsgPayload( const char *array, uint16_t len ) {
        return (packMsgPayloadArrayInter)( reinterpret_cast<const uint8_t *>( array ), len,
                                         DLT_TYPE_INFO_STRG | DLT_SCOD_ASCII );
    }

    template <typename T = void *>
    uint32_t PackMsgPayload( const void *array, uint16_t len ) {
        return packMsgPayloadArrayInter( reinterpret_cast<const uint8_t *>( array ), len,
                                         DLT_TYPE_INFO_RAWD );
    }

    virtual uint32_t PackEnd() = 0;
    virtual void     Reset()   = 0;

    virtual const uint8_t *GetBuffer()     = 0;
    virtual uint32_t       GetBufferSize() = 0;

    // for read
    virtual const uint8_t *GetBufCurPtr()                  = 0;
    virtual const uint8_t *MoveBufCurPtr( uint32_t step )  = 0;
    virtual bool           TestMoveCurPtr( uint32_t step ) = 0;
    virtual bool           ReadEnd()                       = 0;

    template <typename T>
    inline bool ReadBasicValue( T &value ) {
        const uint8_t *ptr_cur = GetBufCurPtr();
        if ( false == TestMoveCurPtr( sizeof( T ) ) ) {
            return false;
        }

        memcpy( reinterpret_cast<uint8_t *>( &value ), ptr_cur, sizeof( T ) );
        MoveBufCurPtr( sizeof( T ) );
        return true;
    }

    template <typename T>
    uint32_t UnPackMsgPayload( T &v ) {
        return (unPackMsgPayloadInter)( reinterpret_cast<uint8_t *>( &v ),
                                      static_cast<uint16_t>( sizeof( T ) ) );
    };

    template <typename T = char *>
    uint32_t UnPackMsgPayload( char *array, uint16_t maxlen ) {
        return UnPackMsgPayloadArrayInter( reinterpret_cast<uint8_t *>( array ), maxlen );
    }

    template <typename T = void *>
    uint32_t UnPackMsgPayload( void *array, uint16_t maxlen ) {
        return (UnPackMsgPayloadArrayInter)( reinterpret_cast<uint8_t *>( array ), maxlen );
    }

    template <typename T = std::string>
    uint32_t UnPackMsgPayload( std::string &v ) {
        std::shared_ptr<uint8_t> sp( new uint8_t[ DLT_LOG_LEN_MAX ],
                                     std::default_delete<uint8_t[]>() );

        uint32_t ret        = UnPackMsgPayloadArrayInter( sp.get(), DLT_LOG_LEN_MAX );
        *( sp.get() + ret ) = '\0';
        v.assign( reinterpret_cast<char *>( sp.get() ), ret );
        return ret;
    }

   protected:
    template <typename T>
    uint32_t GetTypeInfo( DltFormatType type );

    virtual uint32_t packMsgPayloadInter( const uint8_t *buffer, uint16_t len, uint32_t type ) = 0;
    virtual uint32_t packMsgPayloadArrayInter( const uint8_t *buffer, uint16_t len,
                                               uint32_t type )                                 = 0;

    virtual uint32_t unPackMsgPayloadInter( uint8_t *buffer, uint16_t len )      = 0;
    virtual uint32_t UnPackMsgPayloadArrayInter( uint8_t *buffer, uint16_t len ) = 0;
};

template <typename T>
uint32_t DltPayload::GetTypeInfo( DltFormatType type ) {
    uint32_t type_info = 0U;

    if ( std::is_same<T, bool>::value ) {
        type_info = DLT_TYPE_INFO_BOOL;
    } else if ( std::is_same<T, uint8_t>::value ) {
        type_info = DLT_TYPE_INFO_UINT | DLT_TYLE_8BIT;
        if ( DLT_FORMAT_HEX8 == type ) {
            type_info |= DLT_SCOD_HEX;
        } else if ( DLT_FORMAT_BIN8 == type ) {
            type_info |= DLT_SCOD_BIN;
        }
        else {
            // do nothing
        }
    } else if ( std::is_same<T, uint16_t>::value ) {
        type_info = DLT_TYPE_INFO_UINT | DLT_TYLE_16BIT;
        if ( DLT_FORMAT_HEX16 == type ) {
            type_info |= DLT_SCOD_HEX;
        } else if ( DLT_FORMAT_BIN16 == type ) {
            type_info |= DLT_SCOD_BIN;
        }
        else {
            // do nothing
        }
    } else if ( std::is_same<T, uint32_t>::value ) {
        type_info = DLT_TYPE_INFO_UINT | DLT_TYLE_32BIT;
        if ( DLT_FORMAT_HEX32 == type ) {
            type_info |= DLT_SCOD_HEX;
        } else if ( DLT_FORMAT_BIN32 == type ) {
            type_info |= DLT_SCOD_BIN;
        }
        else {
            // do nothing
        }
    } else if ( std::is_same<T, uint64_t>::value ) {
        type_info = DLT_TYPE_INFO_UINT | DLT_TYLE_64BIT;
        if ( DLT_FORMAT_HEX64 == type ) {
            type_info |= DLT_SCOD_HEX;
        } else if ( DLT_FORMAT_BIN64 == type ) {
            type_info |= DLT_SCOD_BIN;
        }
        else {
            // do nothing
        }
    } else if ( std::is_same<T, int8_t>::value ) {
        type_info = DLT_TYPE_INFO_SINT | DLT_TYLE_8BIT;
    } else if ( std::is_same<T, int16_t>::value ) {
        type_info = DLT_TYPE_INFO_SINT | DLT_TYLE_16BIT;
    } else if ( std::is_same<T, int32_t>::value ) {
        type_info = DLT_TYPE_INFO_SINT | DLT_TYLE_32BIT;
    } else if ( std::is_same<T, int64_t>::value ) {
        type_info = DLT_TYPE_INFO_SINT | DLT_TYLE_64BIT;
    } else if ( std::is_same<T, float>::value ) {
        type_info = DLT_TYPE_INFO_FLOA | DLT_TYLE_32BIT;
    } else if ( std::is_same<T, double>::value ) {
        type_info = DLT_TYPE_INFO_FLOA | DLT_TYLE_64BIT;
    } else if ( std::is_same<T, char>::value ) {
        type_info = DLT_TYPE_INFO_STRG | DLT_SCOD_ASCII;
    } else if ( std::is_same<T, void>::value ) {
        type_info = DLT_TYPE_INFO_VARI | DLT_TYPE_INFO_RAWD;
    } else {
        if ( std::is_same<T, bool>::value ) {
            type_info = DLT_TYPE_INFO_BOOL;
        } else if ( std::is_same<T, float>::value ) {
            type_info = DLT_TYPE_INFO_FLOA | DLT_TYLE_32BIT;
        } else if ( std::is_same<T, double>::value ) {
            type_info = DLT_TYPE_INFO_FLOA | DLT_TYLE_64BIT;
        } else if ( std::is_same<T, char>::value ) {
            type_info = DLT_TYPE_INFO_STRG | DLT_SCOD_ASCII;
        } else if ( std::is_same<T, void>::value ) {
            type_info = DLT_TYPE_INFO_STRG | DLT_TYPE_INFO_RAWD;
        } else {
            type_info = DLT_FORMAT_HEX8 | DLT_SCOD_HEX | DLT_TYPE_INFO_RAWD;
        }
    }
    return type_info;
}

}  // namespace dlt
#endif /* __DLT_PAYLOAD_H__ */