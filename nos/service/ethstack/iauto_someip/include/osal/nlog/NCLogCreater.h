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
 * @file NCLogCreater.h
 * @brief
 * @date 2020-06-29
 *
 */
#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#ifndef __NCLOG_CREATER_H__
#define __NCLOG_CREATER_H__

#include <pthread.h>

#include <functional>
#include <map>
#include <mutex>
#include <vector>

#include "osal/nlog/NCCommon.h"

namespace logfilter {
class LogFilterManager;
}

OSAL_BEGIN_NAMESPACE
namespace nlog {
class NCLogger;
class TransportBase;

class NCLogCreater {
   public:
    static NCLogCreater* Ins();
    bool                 RegeditTSyncCallBack( std::function<time_t( void )> func );
    NCLogger*            CreateLogger( const NCString& ctxId, const NCString& ctxDescription,
                                       NCLogLevel ctxDefLogLevel = NCLogLevel::kWarn );
    void                 Start( const NCCONFIGRATION* ncconfig = nullptr );
    bool                 SetDefaultLevel( const NCString& ctxId, NCLogLevel ctxDefLogLevel );
    std::string          GetLogFilePath();
   private:
    NCLogCreater();
    ~NCLogCreater();
    pthread_condattr_t                   m_condattr;
    pthread_mutex_t                      m_mutex;
    pthread_cond_t                       m_cond;
    bool                                 m_isStarted;
    static std::map<NCString, NCLogger*> m_ncloggerMap;
};
}  // namespace nlog
OSAL_END_NAMESPACE
#endif