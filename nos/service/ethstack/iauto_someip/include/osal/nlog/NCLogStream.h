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
 * @file NCLogStream.h
 * @brief
 * @date 2020-06-29
 *
 */
#ifndef __NCLOG_STREAM_H__
#define __NCLOG_STREAM_H__

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include "osal/nlog/NCCommon.h"

namespace dlt {
class DltLogItemPacker;
}

OSAL_BEGIN_NAMESPACE

namespace nlog {

struct NCLogRawBuffer {
    const UINT8 *data;
    UINT8        size;

    // construction
    constexpr NCLogRawBuffer( const UINT8 *data, UINT8 size ) : data( data ), size( size ) {}
};

struct NCLogHex8 {
    UINT8 data;

    // construction
    explicit constexpr NCLogHex8( UINT8 data ) : data( data ) {}
};

struct NCLogHex16 {
    UINT16 data;

    // construction
    explicit constexpr NCLogHex16( UINT16 data ) : data( data ) {}
};

struct NCLogHex32 {
    UINT32 data;

    // construction
    explicit constexpr NCLogHex32( UINT32 data ) : data( data ) {}
};

struct NCLogHex64 {
    UINT64 data;

    // construction
    explicit constexpr NCLogHex64( UINT64 data ) : data( data ) {}
};
struct NCLogBin8 {
    UINT8 data;

    // construction
    explicit constexpr NCLogBin8( UINT8 data ) : data( data ) {}
};

struct NCLogBin16 {
    UINT16 data;

    // construction
    explicit constexpr NCLogBin16( UINT16 data ) : data( data ) {}
};

struct NCLogBin32 {
    UINT32 data;

    // construction
    explicit constexpr NCLogBin32( UINT32 data ) : data( data ) {}
};

struct NCLogBin64 {
    UINT64 data;

    // construction
    explicit constexpr NCLogBin64( UINT64 data ) : data( data ) {}
};

class NCLogger;
class NCLogStream {
   public:
    NCLogStream( const NCLogger *logger, NCLogLevel level,
                 NCLogChannal channel = NCLogChannal::LOG_CHANNAL_MAIN );

    VOID         Flush();
    NCLogStream &Vprintf(const char *format, ...);
    NCLogStream &operator<<( NC_BOOL value ) noexcept;
    NCLogStream &operator<<( UINT8 value ) noexcept;
    NCLogStream &operator<<( UINT16 value ) noexcept;
    NCLogStream &operator<<( UINT32 value ) noexcept;
    NCLogStream &operator<<( UINT64 value ) noexcept;
    NCLogStream &operator<<( INT8 value ) noexcept;
    NCLogStream &operator<<( INT16 value ) noexcept;
    NCLogStream &operator<<( INT32 value ) noexcept;
    NCLogStream &operator<<( INT64 value ) noexcept;
    NCLogStream &operator<<( FLOAT value ) noexcept;
    NCLogStream &operator<<( DOUBLE value ) noexcept;
    NCLogStream &operator<<( NCLogRawBuffer &value ) noexcept;
    NCLogStream &operator<<( NCLogHex8 &value ) noexcept;
    NCLogStream &operator<<( NCLogHex16 &value ) noexcept;
    NCLogStream &operator<<( NCLogHex32 &value ) noexcept;
    NCLogStream &operator<<( NCLogHex64 &value ) noexcept;
    NCLogStream &operator<<( NCLogBin8 &value ) noexcept;
    NCLogStream &operator<<( NCLogBin16 &value ) noexcept;
    NCLogStream &operator<<( NCLogBin32 &value ) noexcept;
    NCLogStream &operator<<( NCLogBin64 &value ) noexcept;
    NCLogStream &operator<<( const CHAR *const value ) noexcept;
    ~NCLogStream();

   private:
    NCLogStream()        = delete;
    NCLogStream &operator=( const NCLogStream &src ) = delete;

    const NCLogger *                   m_logger;
    NCLogChannal                       m_channel;
    dlt::DltLogItemPacker *            m_packer;
    NCLogLevel                         m_level;
    std::shared_ptr<std::stringstream> m_stream;  // android log use
};
}  // namespace nlog
OSAL_END_NAMESPACE
#endif