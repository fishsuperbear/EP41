// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef CSPCCHANNEL_HPP
#define CSPCCHANNEL_HPP

#include "hw_nvmedia_log_impl.h"
#include "hw_nvmedia_eventhandler_common_impl.h"
#include "hw_nvmedia_eventhandler_outputpipeline_impl.h"
#include "CChannel.hpp"

using namespace std;

class CBuffer;
//class CProfiler;
class CPoolManager;
class CClientCommon;
class ICascadedProvider;

/*
 * Single Process Cascaded Channel
 */
class CSPCChannel: public CChannel
{
public:
    CSPCChannel( NvSciBufModule& bufMod, NvSciSyncModule& syncMod, SensorInfo* pSensorInfo, ICascadedProvider* pCascadedProvider, 
        HWNvmediaOutputPipelineContext* i_poutputpipeline);
    ~CSPCChannel();
    SIPLStatus Post( CBuffer* pBuffer );
    // do not use it in the current implement
    SIPLStatus CreateBlocks(HWNvmediaSensorOutputPipelineProfiler* pProfiler, int encodeType);
    SIPLStatus CreateBlocksWithVIC(HWNvmediaSensorOutputPipelineProfiler* pProfiler, int encodeType, void* i_pvicconsumer);
    virtual SIPLStatus Connect();
    virtual SIPLStatus InitBlocks();
    virtual void SetConsumerConfig( const ConsumerConfig& consConfig );
public:
    virtual hw_ret_s32 RegisterDirectCb(struct hw_video_sensorpipelinedatacbconfig_t* i_pcbconfig,
        HWNvmediaEventHandlerRegDataCbConfig* i_peventhandlercbconfig);

protected:
    virtual void GetEventThreadHandlers( bool isStreamRunning, vector<CEventHandler*>& vEventHandlers );

private:

    ICascadedProvider*                m_pCascadedProvider = nullptr;
    unique_ptr<CPoolManager>          m_upPoolManager     = nullptr;
	vector<unique_ptr<CClientCommon>> m_vClients;
    bool                              m_bUseMailbox       = false;
    HWNvmediaOutputPipelineContext* m_poutputpipeline;
};

#endif
