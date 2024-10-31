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
 * @file STObject.cpp
 * @brief implements of STObject
 */


#include "STObject.h"
#include <string.h>
#include "STTaskThread.h"

namespace hozon {
namespace netaos {
namespace sttask {

    STObject::STObject(ST_OBJECT_TYPE objType)
        : m_taskRunner(nullptr)
        , m_objType(objType)
    {
    }

    STObject::~STObject()
    {
    }


    void STObject::setTaskRunner(STTaskRunner* runner)
    {
        m_taskRunner = runner;
    }

    STTaskRunner* STObject::getTaskRunner() const
    {
        return m_taskRunner;
    }

    const ST_OBJECT_TYPE& STObject::getObjectType() const
    {
        return m_objType;
    }

    std::string STObject::toString()
    {
        std::string val;
        char buf[128] = { 0 };
        snprintf(buf, sizeof(buf), "%s[%p]", getObjectName().c_str(), this);
        val.assign(buf, strlen(buf));
        return val;
    }

    std::string STObject::getObjectName()
    {
        return std::string("STObject");
    }

} // end of sttask
} // end of netaos
} // end of hozon
/* EOF */