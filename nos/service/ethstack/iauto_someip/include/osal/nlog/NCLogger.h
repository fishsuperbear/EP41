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
 * @file NCLogger.h
 * @brief
 * @date 2020-06-29
 *
 */
#ifndef __NCLOG_LOGGER_H__
#define __NCLOG_LOGGER_H__

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include <mutex>

#include "osal/nlog/NCCommon.h"
#include "osal/nlog/NCLogStream.h"

OSAL_BEGIN_NAMESPACE
namespace nlog {
class NCLogger {
   public:
    ~NCLogger();
    NCLogger( NCString ctxId, NCString ctxDescription,
              NCLogLevel ctxDefLogLevel = NCLogLevel::kWarn );
    NCLogStream LogFatal( NCLogChannal channel = NCLogChannal::LOG_CHANNAL_MAIN );
    NCLogStream LogError( NCLogChannal channel = NCLogChannal::LOG_CHANNAL_MAIN );
    NCLogStream LogWarn( NCLogChannal channel = NCLogChannal::LOG_CHANNAL_MAIN );
    NCLogStream LogInfo( NCLogChannal channel = NCLogChannal::LOG_CHANNAL_MAIN );
    NCLogStream LogDebug( NCLogChannal channel = NCLogChannal::LOG_CHANNAL_MAIN );
    NCLogStream LogVerbose( NCLogChannal channel = NCLogChannal::LOG_CHANNAL_MAIN );
    void* getLogBuffer() const;

    NCString   GetCtxID() const;
    NCString   GetDesp() const;
    NCLogLevel GetLoggerLevel() const;

    NC_BOOL SetDefaultLevel( NCLogLevel ctxDefLogLevel );

   private:
    NCString   m_ctxid;
    NCString   m_ctxdesp;
    NCLogLevel m_level;
    std::mutex mtx;
    void*      m_buffer;
};
}  // namespace nlog
OSAL_END_NAMESPACE
#endif