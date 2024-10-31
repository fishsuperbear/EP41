// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef CEVENTHANDLER_H
#define CEVENTHANDLER_H

#include "nvscistream.h"
#include <cstring>

enum EventStatus
{
    EVENT_STATUS_OK = 0,
    EVENT_STATUS_COMPLETE,
    EVENT_STATUS_TIMED_OUT,
    EVENT_STATUS_ERROR
};

class CEventHandler {
public:
    CEventHandler() = delete;
    CEventHandler(std::string name, NvSciStreamBlock handle, uint32_t uSensor)
    {
       m_name = name + std::to_string(uSensor);
       m_handle = handle;
       m_uSensorId = uSensor;
    }
    ~CEventHandler(void) {};

    virtual EventStatus HandleEvents(void) = 0;

    NvSciStreamBlock GetHandle(void)
    {
        return m_handle;
    }

    std::string GetName(void)
    {
        return m_name;
    }

protected:
    uint32_t m_uSensorId;
    NvSciStreamBlock m_handle = 0U;
    std::string m_name;
};

#endif
