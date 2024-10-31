/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/* STL Headers */
#include <unistd.h>
#include <cstring>
#include <iostream>

#include "nvscibuf.h"
#include "nvscisync.h"
#include "nvscistream.h"
#include "NvSIPLDeviceBlockInfo.hpp"
#include "CCascadedMaster.hpp"
#include "CSPCChannel.hpp"
#include "ICascadedProvider.hpp"
#include "CBuffer.hpp"
#include "CPoolManager.hpp"

using namespace std;

/** CCascadedMaster class */
CCascadedMaster::CCascadedMaster( NvSciBufModule sciBufModule,
                                  NvSciSyncModule sciSyncModule,
                                  SensorInfo* pSensorInfo,
                                  ICascadedProvider* pCascadedProvider,
                                  const ConsumerConfig& consConfig )
: m_sciBufModule( sciBufModule ),
  m_sciSyncModule( sciSyncModule ),
  m_pSensorInfo( pSensorInfo ),
  m_pCascadedProvider( pCascadedProvider ),
  m_consumerConfig( consConfig )
{
}

CCascadedMaster::~CCascadedMaster()
{
}

SIPLStatus CCascadedMaster::Prepare(HWNvmediaOutputPipelineContext* i_poutputpipeline,int encodeType,void* i_pvicconsumer)
{
    LOG_DBG( "CCascadedMaster: start.\n" );
    m_upSPCChannel.reset( new CSPCChannel( m_sciBufModule, m_sciSyncModule, m_pSensorInfo, m_pCascadedProvider, i_poutputpipeline) );
    CHK_PTR_AND_RETURN( m_upSPCChannel, "new CSPCChannel" );

    auto status = m_upSPCChannel->CreateBlocksWithVIC(nullptr,encodeType, i_pvicconsumer);
    CHK_STATUS_AND_RETURN( status, "CCascadedMaster: Create blocks." );

    m_upSPCChannel->SetConsumerConfig( m_consumerConfig );

    status = m_upSPCChannel->Connect();
    CHK_STATUS_AND_RETURN( status, "CCascadedMaster: Channel connect." );

    status = m_upSPCChannel->InitBlocks();
    CHK_STATUS_AND_RETURN( status, "CCascadedMaster InitBlocks" );

    status = m_upSPCChannel->Reconcile();
    CHK_STATUS_AND_RETURN( status, "CCascadedMaster Channel Reconcile" );

    return NVSIPL_STATUS_OK;
}

void CCascadedMaster::Start()
{
    LOG_DBG( "CCascadedMaster::Start.\n" );
    if ( m_upSPCChannel.get() )
    {
        m_upSPCChannel->Start();
    }
}

void CCascadedMaster::Stop()
{
    LOG_DBG( "CCascadedMaster: stop.\n" );
    if ( m_upSPCChannel.get() )
    {
        m_upSPCChannel->Stop();
    }
}

void CCascadedMaster::OnFrameAvailable( CBuffer* pBuffer )
{
    LOG_DBG( "CCascadedMaster: OnFrameAvailable.\n" );
    if ( m_upSPCChannel.get() )
    {
        CSPCChannel* pSPCChannel = dynamic_cast<CSPCChannel*>( m_upSPCChannel.get() );
        pSPCChannel->Post( pBuffer );
    }
}

hw_ret_s32 CCascadedMaster::RegisterDirectCb(struct hw_video_sensorpipelinedatacbconfig_t* i_pcbconfig,
    HWNvmediaEventHandlerRegDataCbConfig* i_peventhandlercbconfig) 
{
    LOG_DBG("CCascadedMaster: RegisterDirectCb.\n");
    if (m_upSPCChannel.get())
    {
        CSPCChannel* pSPCChannel = dynamic_cast<CSPCChannel*>(m_upSPCChannel.get());
        pSPCChannel->RegisterDirectCb(i_pcbconfig, i_peventhandlercbconfig);
    }
    return 0;
}


    

