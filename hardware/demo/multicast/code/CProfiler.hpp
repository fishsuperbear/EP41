/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <mutex>
#include <condition_variable>

#include "NvSIPLClient.hpp"

#ifndef CPROFILER_HPP
#define CPROFILER_HPP

using namespace nvsipl;

class CProfiler
{
 public:
    typedef struct {
        std::mutex profDataMut;
        uint64_t uFrameCount;
        uint64_t uPrevFrameCount;
    } ProfilingData;

    void Init(uint32_t uSensor, INvSIPLClient::ConsumerDesc::OutputType outputType)
    {
        m_uSensor = uSensor;
        m_outputType = outputType;

        m_profData.profDataMut.lock();
        m_profData.uFrameCount = 0U;
        m_profData.uPrevFrameCount = 0U;
        m_profData.profDataMut.unlock();
    }

    void OnFrameAvailable(void)
    {
        m_profData.profDataMut.lock();
        m_profData.uFrameCount++;
        m_profData.profDataMut.unlock();
    }

    ~CProfiler()
    {
    }

    uint32_t m_uSensor = UINT32_MAX;
    INvSIPLClient::ConsumerDesc::OutputType m_outputType;
    ProfilingData m_profData;
};

#endif // CPROFILER_HPP
