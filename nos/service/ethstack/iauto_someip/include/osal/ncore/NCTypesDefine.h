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
 * @file NCTypesDefine.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCTYPESDEFINE_H_
#define INCLUDE_NCORE_NCTYPESDEFINE_H_

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <utime.h>
#include <wchar.h>

#include <cstring>
#include <type_traits>

#ifdef __ILP32__
#define __WORDSIZE 32
#endif

#ifdef __ILP64__
#define __WORDSIZE 64
#endif

#include "osal/ncore/NCGlobalAPI.h"
#include "osal/ncore/NCNameSpace.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE
//////////////////////////////////////////////////////////
/// Basical types define
typedef char          CHAR;
typedef signed char   SCHAR;
typedef unsigned char UCHAR;

typedef int8_t  INT8;
typedef uint8_t UINT8;

typedef int16_t  INT16;
typedef uint16_t UINT16;

typedef char16_t CHAR16;
typedef UINT16   UCHAR16;
typedef wchar_t  WCHAR32;

typedef int32_t  INT32;
typedef uint32_t UINT32;

typedef int64_t  INT64;
typedef uint64_t UINT64;

typedef float  FLOAT;
typedef double DOUBLE;

typedef bool NC_BOOL;

typedef void   VOID;
typedef UINT32 NC_ERROR;       // < used define error code
typedef UINT64 NC_TIME_T;      // < used define time
typedef UINT64 NC_FILESIZE_T;  // < used define file size
typedef INT64  NC_FILEOFF_T;   // < used define offset in file
typedef UINT32 NC_BUF_T;       // < used define buffer size
                               // < enmu is type define UINT32 normaly

const NC_ERROR NC_NOERROR = 0;

const NC_BOOL NC_TRUE  = true;
const NC_BOOL NC_FALSE = false;

#ifndef MIN
#if ( defined linux ) || ( defined __linux__ )
#define MIN( a, b ) ( ( a ) < ( b ) ) ? ( a ) : ( b )
#define MAX( a, b ) ( ( a ) > ( b ) ) ? ( a ) : ( b )
#else
#define MIN __min
#define MAX __max
#endif
#endif

#ifndef MAXUINT16
static const UINT16 MAXUINT16 = 65535U;
#endif

#ifndef MAXUINT32
static const UINT32 MAXUINT32 = 4294967295U;
#endif

#ifndef MAXUINT64
static const UINT64 MAXUINT64 = 0xFFFFFFFFFFFFFFFFUL;
#endif

#ifndef INFINITE32
static const UINT32 INFINITE32 = MAXUINT32;
#endif

#ifndef INFINITE64
static const UINT64 INFINITE64 = MAXUINT64;
#endif

#ifndef MAX_PATH
static const UINT32 MAX_PATH = 1024U;
#endif

#ifndef PAI
static const FLOAT PAI = 3.141592654F;
#endif

#define NCGETPID ( (long) getpid() )

// Define for payload enumeration serialization
template <typename Enumeration>
static inline auto NCTypesEnumToInteger( Enumeration V ) ->
    typename std::underlying_type<Enumeration>::type {
    return ( static_cast<typename std::underlying_type<Enumeration>::type>( V ) );
}

template <typename Enumeration, typename Integer>
static inline auto NCTypesIntegerToEnum( Integer V ) -> decltype( static_cast<Enumeration>( V ) ) {
    return ( static_cast<Enumeration>( V ) );
}

// global functions
#ifdef __cplusplus
extern "C" {
#endif
__attribute__( ( visibility( "default" ) ) ) void msleepTime( NC_TIME_T msecs );
#ifdef __cplusplus
}
#endif
OSAL_END_NAMESPACE

// global functions
#ifdef __cplusplus
extern "C" {
#endif
#if ( defined linux ) || ( defined __linux__ )
#include <syscall.h>
__attribute__( ( visibility( "default" ) ) ) pid_t gettid();
#endif
#ifdef __cplusplus
}
#endif

#endif  // INCLUDE_NCORE_NCTYPESDEFINE_H_
/* EOF */
