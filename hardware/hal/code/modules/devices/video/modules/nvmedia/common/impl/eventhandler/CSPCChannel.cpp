// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include "CChannel.hpp"
#include "CFactory.hpp"
#include "CPoolManager.hpp"
#include "CProducer.hpp"
#include "CConsumer.hpp"
#include "CClientCommon.hpp"
#include "CSPCChannel.hpp"
#include "ICascadedProvider.hpp"

using namespace std;

CSPCChannel::CSPCChannel( NvSciBufModule& bufMod, NvSciSyncModule& syncMod, SensorInfo* pSensorInfo, ICascadedProvider* pCascadedProvider, HWNvmediaOutputPipelineContext* i_poutputpipeline)
: CChannel( "SPCChannel", bufMod, syncMod, pSensorInfo ),
  m_pCascadedProvider( pCascadedProvider ),
    m_poutputpipeline(i_poutputpipeline)
{
}

CSPCChannel::~CSPCChannel()
{
    PLOG_DBG( "Release.\n" );

    if ( m_upPoolManager != nullptr && m_upPoolManager->GetHandle() != 0U )
    {
        NvSciStreamBlockDelete( m_upPoolManager->GetHandle() );
    }

    if ( m_vClients[0] != nullptr && m_vClients[0]->GetHandle() != 0U )
    {
        NvSciStreamBlockDelete( m_vClients[0]->GetHandle() );
    }

    for ( uint32_t i = 1U; i < m_vClients.size(); i++ )
    {
        CConsumer* pConsumer = dynamic_cast<CConsumer*>( m_vClients[i].get() );
        if ( pConsumer != nullptr && pConsumer->GetHandle() != 0U )
        {
            NvSciStreamBlockDelete( pConsumer->GetHandle() );
        }

        if ( pConsumer != nullptr && pConsumer->GetQueueHandle() != 0U )
        {
            NvSciStreamBlockDelete( pConsumer->GetQueueHandle() );
        }
    }
}

SIPLStatus CSPCChannel::Post( CBuffer* pBuffer )
{
    PLOG_DBG( "Post\n" );

    CProducer* pProducer = dynamic_cast<CProducer*>( m_vClients[0].get() );
    PCHK_PTR_AND_RETURN( pProducer, "m_vClients[0] converts to CProducer" );

    auto status = pProducer->Post( pBuffer );
    PCHK_STATUS_AND_RETURN( status, "Post" );

    return NVSIPL_STATUS_OK;
}

SIPLStatus CSPCChannel::CreateBlocks(HWNvmediaSensorOutputPipelineProfiler* pProfiler, int encodeType) {
    PLOG_ERR("CreateBlocks call the wrong function!\n");

    return NVSIPL_STATUS_BAD_ARGUMENT;
}

SIPLStatus CSPCChannel::CreateBlocksWithVIC(HWNvmediaSensorOutputPipelineProfiler* pProfiler, int encodeType, void* i_pvicconsumer) {
    PLOG_DBG( "CreateBlocks.\n" );
    UseCaseInfo ucInfo;
    ucInfo.isMultiElems = false;
    ucInfo.isEnableICP = false;

    m_upPoolManager = CFactory::CreatePoolManager( m_pSensorInfo->id, MAX_NUM_PACKETS );
    CHK_PTR_AND_RETURN( m_upPoolManager, "CFactory::CreatePoolManager." );
    PLOG_DBG( "VIC PoolManager is created.\n" );

    std::unique_ptr<CProducer> upProducer = CFactory::CreateVICProducer( m_upPoolManager->GetHandle(), m_pSensorInfo->id, m_pCascadedProvider,ucInfo );
    PCHK_PTR_AND_RETURN( upProducer, "CFactory::CreateVICProducer." );
    PLOG_DBG( "VIC Producer is created.\n" );

    upProducer->SetProfiler( pProfiler );
    m_vClients.push_back( std::move( upProducer ) );

    std::unique_ptr<CConsumer> upEncConsumer = CFactory::CreateConsumer(ENC_CONSUMER, m_poutputpipeline->psensorinfo,
        m_poutputpipeline->poutputpipeline_ops->outputtype, m_bUseMailbox, ucInfo, m_poutputpipeline, encodeType, i_pvicconsumer);
    PCHK_PTR_AND_RETURN( upEncConsumer, "CFactory::Create COMMON_CONSUMER consumer" );
    HW_NVMEDIA_LOG_DEBUG("Encoder consumer is created.\r\n");

    m_vClients.push_back( std::move( upEncConsumer ) );

    PLOG_DBG( "Encoder consumer is created.\n" );

    return NVSIPL_STATUS_OK;
}

SIPLStatus CSPCChannel::Connect()
{
    NvSciStreamEventType event;

    PLOG_DBG( "Connect.\n" );

    auto sciErr = NvSciStreamBlockConnect( m_vClients[0]->GetHandle(), m_vClients[1]->GetHandle() );
    PCHK_NVSCISTATUS_AND_RETURN( sciErr, ( "VIC Producer connect to" + m_vClients[1]->GetName() ).c_str() );

    LOG_INFO("Connecting to the stream...\n");

    /*
     * query producer
     */
    sciErr = NvSciStreamBlockEventQuery( m_vClients[0]->GetHandle(), QUERY_TIMEOUT_FOREVER, &event );
    PCHK_NVSCICONNECT_AND_RETURN( sciErr, event, "VIC producer" );
    PLOG_DBG( "VIC Producer is connected.\n" );

    sciErr = NvSciStreamBlockEventQuery( m_upPoolManager->GetHandle(), QUERY_TIMEOUT_FOREVER, &event );
    PCHK_NVSCICONNECT_AND_RETURN( sciErr, event, "VIC pool" );
    PLOG_DBG( "VIC Pool is connected.\n" );

    /*
     * query consumers and queues
     */
    for ( uint32_t i = 1U; i < m_vClients.size(); i++ )
    {
        CConsumer* pConsumer = dynamic_cast<CConsumer*>( m_vClients[i].get() );
        sciErr = NvSciStreamBlockEventQuery( pConsumer->GetQueueHandle(), QUERY_TIMEOUT_FOREVER, &event );
        PCHK_NVSCICONNECT_AND_RETURN( sciErr, event, "Enc queue" );
        PLOG_DBG( "Enc Queue:%u is connected.\n", ( i - 1 ) );

        sciErr = NvSciStreamBlockEventQuery( pConsumer->GetHandle(), QUERY_TIMEOUT_FOREVER, &event );
        PCHK_NVSCICONNECT_AND_RETURN( sciErr, event, "Enc consumer" );
        PLOG_DBG( "Enc Consumer:%u is connected.\n", ( i - 1 ) );
    }

    LOG_INFO( "All VIC->ENC blocks are connected to the stream!\n" );
    return NVSIPL_STATUS_OK;
}

SIPLStatus CSPCChannel::InitBlocks()
{
    PLOG_DBG( "InitBlocks.\n" );

    auto status = m_upPoolManager->Init();
    PCHK_STATUS_AND_RETURN( status, "Pool Init" );

    for ( auto& upClient: m_vClients )
    {
        auto status = upClient->Init( m_bufModule, m_syncModule );
        PCHK_STATUS_AND_RETURN( status, ( upClient->GetName() + " Init" ).c_str() );
    }

    return NVSIPL_STATUS_OK;
}

void CSPCChannel::SetConsumerConfig( const ConsumerConfig& consConfig )
{
    for ( uint32_t i = 1U; i < m_vClients.size(); i++ )
    {
        CConsumer* pConsumer = dynamic_cast<CConsumer*>( m_vClients[i].get() );
        if ( pConsumer != nullptr )
        {
            pConsumer->SetConsumerConfig( consConfig );
        }
    }
    m_bUseMailbox = consConfig.bUseMailbox;
}

void CSPCChannel::GetEventThreadHandlers( bool isStreamRunning, std::vector<CEventHandler*>& vEventHandlers )
{
    if ( !isStreamRunning )
    {
        vEventHandlers.push_back( m_upPoolManager.get() );
    }

    for ( auto& upClient: m_vClients )
    {
        vEventHandlers.push_back( upClient.get() );
    }
}

hw_ret_s32 CSPCChannel::RegisterDirectCb(struct hw_video_sensorpipelinedatacbconfig_t* i_pcbconfig,
    HWNvmediaEventHandlerRegDataCbConfig* i_peventhandlercbconfig)
{
    /*
    * Only support direct mode.
    */
    if (!i_peventhandlercbconfig->bdirectcb)
    {
        HW_NVMEDIA_LOG_ERR("Only support direct data cb mode!\r\n");
        return HW_RET_S32_HW_HAL_NVMEDIA(HW_RET_S32_CODE_HW_HAL_NVMEDIA_ONLY_SUPPORT_DIRECTCB_MODE);
    }
    for (uint32_t i = 1U; i < m_vClients.size(); i++)
    {
        CConsumer* pConsumer = dynamic_cast<CConsumer*>(m_vClients[i].get());
        //if (pConsumer != nullptr && pConsumer->GetConsumerType() == i_peventhandlercbconfig->consumertype)
	if (pConsumer != nullptr)
        {
            CHK_LOG_SENTENCE_HW_RET_S32(pConsumer->RegisterDirectCb(i_pcbconfig, i_peventhandlercbconfig));
        }
    }
    //return m_upCascadedMaster->RegisterDirectCb(i_pcbconfig, i_peventhandlercbconfig);
    return 0;
}
