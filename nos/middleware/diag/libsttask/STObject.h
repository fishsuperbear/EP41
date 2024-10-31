/**
 * Copyright @ 2021 - 2023 Hozon Auto Co., Ltd.
 * All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are NOT permitted except as agreed by
 * Hozon Auto Co., Ltd.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
/**
 * @file  STObject.h
 * @brief Class of STObject
 */
#ifndef STOBJECT_H
#define STOBJECT_H
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include <stdio.h>
#include <string.h>
#include <string>
#include "STObjectDef.h"

#define CAST_TASK_CB(fun) (static_cast<STObject::TaskCB>(fun))
#define CAST_MONITOR_CB(fun) (static_cast<STObject::MonitorCB>(fun))
#define CAST_CALL_CB(fun) (static_cast<STObject::CallCB>(fun))
#define CAST_EVENTWATCHER_CB(fun) (static_cast<STObject::EventWatcherCB>(fun))
#define CAST_PERIOD_CB(fun) (static_cast<STObject::PeriodCB>(fun))

namespace hozon {
namespace netaos {
namespace sttask {

    class STEvent;
    class STTask;
    class STCall;
    class STTaskRunner;
    /**
     * @brief Class of STObject
     *
     * This class is a normal task.
     */
    class STObject
    {
    public:
        typedef void (STObject::*TaskCB)(STTask*,  uint32_t);
        typedef void (STObject::*MonitorCB)(STEvent*);
        typedef void (STObject::*CallCB)(STCall*);
        typedef bool (STObject::*EventWatcherCB)(bool, STEvent*);
        typedef void (STObject::*PeriodCB)(uint32_t,  uint32_t);

        STObject(ST_OBJECT_TYPE objType);
        virtual ~STObject();

        void                    setTaskRunner(STTaskRunner* runner);
        STTaskRunner*           getTaskRunner() const;

        const ST_OBJECT_TYPE&   getObjectType() const;
        virtual std::string     toString();
        virtual std::string     getObjectName();

    private:
        STTaskRunner* m_taskRunner;
        ST_OBJECT_TYPE m_objType;

    private:
        STObject(const STObject&);
        STObject& operator=(const STObject&);
    };

} // end of sttask
} // end of netaos
} // end of hozon

#endif /* STOBJECT_H */
/* EOF */