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
 * @file NCDump.h
 * @brief
 * @date 2021-02-08
 *
 */
#ifndef __NC_DUMP_H__
#include <osal/ncore/NCRefBase.h>
#include <osal/ncore/NCString.h>

#include <map>
#include <string>
#include <vector>

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE
class NCDumpItem {
   public:
    NCDumpItem( const NCString& type, const NCString& cmd, const NCString& help, NC_BOOL iswatch )
        : mType( type ), mCommand( cmd ), mHelp( help ), mIsWatch( iswatch ) {}
    virtual ~NCDumpItem()                                               = default;
    virtual NCString OnCommand( const std::vector<NCString>& commands ) = 0;

   public:
    NCString mType;
    NCString mCommand;
    NCString mHelp;
    NC_BOOL  mIsWatch;
};

class NCWatchDumpItem : public NCDumpItem {
   public:
    NCWatchDumpItem( const NCString& type, const NCString& cmd, const NCString& help )
        : NCDumpItem( type, cmd, help, NC_TRUE ) {}
    virtual NCString OnWatchCommand( const std::vector<NCString>& commands ) = 0;
    virtual NCString OnCommand( const std::vector<NCString>& commands ) final {
        return OnWatchCommand( commands );
    }
};

class NCOperateDumpItem : public NCDumpItem {
   public:
    NCOperateDumpItem( const NCString& type, const NCString& cmd, const NCString& help )
        : NCDumpItem( type, cmd, help, NC_FALSE ) {}
    virtual NCString OnOperateCommand( const std::vector<NCString>& commands ) = 0;
    virtual NCString OnCommand( const std::vector<NCString>& commands ) final {
        return OnOperateCommand( commands );
    }
};

class NCDump {
   public:
    static NCDump* Ins();

    virtual VOID RegeditItem( sp<NCDumpItem> item ) = 0;

   protected:
    NCDump()          = default;
    virtual ~NCDump() = default;
};
OSAL_END_NAMESPACE
#endif
