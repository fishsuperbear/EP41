/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef CCASCADEDMASTER_HPP
#define CCASCADEDMASTER_HPP

/* STL Headers */
#include <unistd.h>
#include <cstring>
#include <iostream>

#include "hw_nvmedia_eventhandler_common_impl.h"

#include "nvscibuf.h"
#include "nvscisync.h"
#include "NvSIPLDeviceBlockInfo.hpp"
//#include "CUtils.hpp"
#include "hw_nvmedia_eventhandler_common_impl.h"

using namespace std;
using namespace nvsipl;

class ICascadedProvider;
class CChannel;
class CBuffer;

/** CCascadedMaster class */
class CCascadedMaster
{
 public:
    CCascadedMaster( NvSciBufModule sciBufModule,
                     NvSciSyncModule sciSyncModule,
                     SensorInfo* pSensorInfo,
                     ICascadedProvider* pCascadedProvider,
                     const ConsumerConfig& consConfig );
    ~CCascadedMaster();
    SIPLStatus Prepare(HWNvmediaOutputPipelineContext* i_poutputpipeline, int encodeType, void* i_pvicconsumer);
    void Start();
    void Stop();
    void OnFrameAvailable( CBuffer* pBuffer );
    void SetUsecaseInfo(UseCaseInfo &ucinfo)
    {
        m_ucInfo = ucinfo;
    };

public:
    virtual hw_ret_s32 RegisterDirectCb(struct hw_video_sensorpipelinedatacbconfig_t* i_pcbconfig,
        HWNvmediaEventHandlerRegDataCbConfig* i_peventhandlercbconfig);

private:
    NvSciBufModule        m_sciBufModule;
    NvSciSyncModule       m_sciSyncModule;
    SensorInfo*     m_pSensorInfo;
    ICascadedProvider*    m_pCascadedProvider;
    const ConsumerConfig  m_consumerConfig;
    unique_ptr<CChannel>  m_upSPCChannel;
    UseCaseInfo m_ucInfo;
};

#endif //CCASCADEDMASTER_HPP
