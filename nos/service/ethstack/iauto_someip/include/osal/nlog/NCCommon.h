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
 * @file NCCommon.h
 * @brief
 * @date 2020-06-29
 *
 */

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#ifndef __NCCOMMON_H__
#define __NCCOMMON_H__

#include <osal/ncore/NCString.h>
#include <osal/ncore/NCTypesDefine.h>

#include <string>
#include <map>

OSAL_BEGIN_NAMESPACE
namespace nlog {
#if !defined( PACKED )
#define PACKED __attribute__( ( aligned( 1 ), packed ) )
#endif

typedef enum : uint8_t {
    kOff     = 0x00,  // No logging
    kFatal   = 0x01,  // Fatal error, not recoverable
    kError   = 0x02,  // Error with impact to correct functionality
    kWarn    = 0x03,  // Warning if correct behavior cannot be ensured
    kInfo    = 0x04,  // Informational, providing high level understanding
    kDebug   = 0x05,  // Detailed information for programmers
    kVerbose = 0x06   // Extra-verbose debug messages (highest grade of information)
} NCLogLevel;

typedef enum : uint8_t {
    LOG_CHANNAL_MAIN   = 0x00, /**< channal main */
    LOG_CHANNAL_SYSTEM = 0x01, /**< channal system */
    LOG_CHANNAL_EVENT  = 0x02, /**< channal event */
    LOG_CHANNAL_RADIO  = 0x03, /**< channal radio */
    LOG_CHANNAL_MAX            /**< maximum value, used for range check */
} NCLogChannal;

// @brief Log mode. Flags, used to configure the sink for log messages
// In order to combine flags, at least the OR and AND operators needs to be provided for this type

typedef enum : uint8_t {
    kNone    = 0x00,  // no log
    kRemote  = 0x01,  // Sent remotely
    kFile    = 0x02,  // Save to file
    kConsole = 0x04,  // Forward to console
    kStdout  = 0x08   // Forward to stdout
} NCLogMode;

typedef struct {
    char    pattern[ 4 ];
    int32_t serviceid;
    int32_t length;
} PACKED SerHead;

typedef struct {
    uint8_t pattern[ 4 ];
    int32_t len;
} PACKED MSGHEAD;

typedef struct {
    uint8_t* buffer;
    int32_t  start;
    int32_t  length;
} PACKED MSG;

typedef struct {
    uint32_t write;
    uint32_t read;
} PACKED BUFFERHEAD;

typedef struct {
    std::string ecuid;
    std::string appid;
    std::string desc;
    std::string filepath;
    std::string timebase;
    uint64_t    filesize;
    uint32_t    mLogTransport;
    uint32_t    maxfilenum;
    uint32_t    maxmsgnum;
    NCLogLevel  appdefaultlevel;
    NCLogMode   mode;
    bool        mStdOutFlg;
    std::map<std::string,NCLogLevel> tagFilterList;
} NCCONFIGRATION;

inline NCLogMode operator|( NCLogMode lhs, NCLogMode rhs ) {
    return static_cast<NCLogMode>( static_cast<uint8_t>( lhs ) | static_cast<uint8_t>( rhs ) );
}

inline NCLogMode operator&( NCLogMode lhs, NCLogMode rhs ) {
    return static_cast<NCLogMode>( static_cast<uint8_t>( lhs ) & static_cast<uint8_t>( rhs ) );
}
}  // namespace nlog
OSAL_END_NAMESPACE
#endif
