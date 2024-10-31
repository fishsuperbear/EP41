// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include <algorithm>
#include "CVICProducer.hpp"
#include "ICascadedProvider.hpp"
#include "CPoolManager.hpp"
#include "CBuffer.hpp"
constexpr static int32_t OUTPUT_TYPE_UNDEFINED = -1;


CVICProducer::CVICProducer( NvSciStreamBlock handle, uint32_t uSensor, ICascadedProvider* pCascadedProvider )
: CProducer( "CVICProducer", handle, uSensor,nullptr ),
  m_pCascadedProvider( pCascadedProvider ),
  m_vBufObjs( MAX_NUM_PACKETS, nullptr )
{
    //memset(m_elemTypeToOutputType, OUTPUT_TYPE_UNDEFINED, sizeof(m_elemTypeToOutputType));
    //memset(m_outputTypeToElemType, ELEMENT_TYPE_UNDEFINED, sizeof(m_outputTypeToElemType));
}

CVICProducer::~CVICProducer()
{
    PLOG_DBG( "Release.\n" );
    for ( NvSciBufObj& bufObj : m_vBufObjs )
    {
        if ( bufObj != nullptr )
        {
            NvSciBufObjFree( bufObj );
            bufObj = nullptr;
        }
    }
}

SIPLStatus CVICProducer::HandleClientInit()
{
    return NVSIPL_STATUS_OK;
}

// Create and set CPU signaler and waiter attribute lists.
SIPLStatus CVICProducer::SetSyncAttrList(PacketElementType userType,
                                          NvSciSyncAttrList &signalerAttrList,
                                          NvSciSyncAttrList &waiterAttrList)
{
    NvSciSyncAccessPerm signalerCpuPerm = NvSciSyncAccessPerm_SignalOnly;
    bool needCpuAccess = true;
    NvSciSyncAttrKeyValuePair signalerKeyValuePair[] =
    {
        { NvSciSyncAttrKey_NeedCpuAccess, &needCpuAccess, sizeof( needCpuAccess )     },
        { NvSciSyncAttrKey_RequiredPerm,  &signalerCpuPerm, sizeof( signalerCpuPerm ) }
    };
    auto sciErr = NvSciSyncAttrListSetAttrs( signalerAttrList, signalerKeyValuePair, sizeof( signalerKeyValuePair ) / sizeof( signalerKeyValuePair[0] ) );
    PCHK_NVSCISTATUS_AND_RETURN( sciErr, "Signaler NvSciSyncAttrListSetAttrs" );

    NvSciSyncAccessPerm waiterCpuPerm = NvSciSyncAccessPerm_WaitOnly;
    NvSciSyncAttrKeyValuePair waiterKeyValuePair[] =
    {
        { NvSciSyncAttrKey_NeedCpuAccess, &needCpuAccess, sizeof( needCpuAccess )     },
        { NvSciSyncAttrKey_RequiredPerm,  &waiterCpuPerm, sizeof( waiterCpuPerm ) }
    };
    sciErr = NvSciSyncAttrListSetAttrs( waiterAttrList, waiterKeyValuePair, sizeof( waiterKeyValuePair ) / sizeof( waiterKeyValuePair[0] ) );
    PCHK_NVSCISTATUS_AND_RETURN( sciErr, "CPU waiter NvSciSyncAttrListSetAttrs" );

    return NVSIPL_STATUS_OK;
}

// Buffer setup functions
SIPLStatus CVICProducer::SetDataBufAttrList(PacketElementType userType, NvSciBufAttrList &bufAttrList)
{
    auto status = m_pCascadedProvider->GetNvSciBufAttrList( bufAttrList );
    PCHK_STATUS_AND_RETURN( status, "CascadedProvider GetNvSciBufAttrList" );

    return NVSIPL_STATUS_OK;
}

// Create client buffer objects from NvSciBufObj
SIPLStatus CVICProducer::MapDataBuffer( PacketElementType userType, uint32_t packetIndex, NvSciBufObj bufObj )
{
    PLOG_DBG( "Mapping data buffer, packetIndex: %u.\n", packetIndex );
    auto sciErr = NvSciBufObjDup( bufObj, &m_vBufObjs[packetIndex] );
    PCHK_NVSCISTATUS_AND_RETURN( sciErr, "NvSciBufObjDup" );

    return NVSIPL_STATUS_OK;
}

// Create client buffer objects from NvSciBufObj
SIPLStatus CVICProducer::MapMetaBuffer( uint32_t packetIndex, NvSciBufObj bufObj )
{
    return NVSIPL_STATUS_OK;
}

SIPLStatus CVICProducer::RegisterSignalSyncObj(PacketElementType userType, NvSciSyncObj signalSyncObj)
{
    return NVSIPL_STATUS_OK;
}

SIPLStatus CVICProducer::RegisterWaiterSyncObj(PacketElementType userType, NvSciSyncObj waiterSyncObj)
{
    return NVSIPL_STATUS_OK;
}

SIPLStatus CVICProducer::HandleSetupComplete()
{
    auto status = CProducer::HandleSetupComplete();
    PCHK_STATUS_AND_RETURN( status, "HandleSetupComplete" );

    PLOG_DBG( "RegisterNvSciBufObjs\n" );
    status = m_pCascadedProvider->RegisterNvSciBufObjs( m_vBufObjs );
    PCHK_STATUS_AND_RETURN( status, "CascadedProvider::RegisterNvSciBufObjs" );

    return NVSIPL_STATUS_OK;
}

//Before calling PreSync, m_nvmBuffers[packetIndex] should already be filled.
SIPLStatus CVICProducer::InsertPrefence( PacketElementType userType, uint32_t packetIndex, NvSciSyncFence& prefence )
{
    return NVSIPL_STATUS_OK;
}

SIPLStatus CVICProducer::GetPostfence( INvSIPLClient::ConsumerDesc::OutputType outputType,
                                       uint32_t packetIndex,
                                       NvSciSyncFence *pPostfence )
{
    return NVSIPL_STATUS_OK;
}

void CVICProducer::OnPacketGotten( uint32_t packetIndex )
{
    if ( m_pBuffers[packetIndex] != nullptr) {
            //m_siplBuffers[i].nvmBuffers[packetIndex]->Release();
	    m_pBuffers[packetIndex]->Release();
        }
}

SIPLStatus CVICProducer::GetPacketId(std::vector<NvSciBufObj> bufObjs, NvSciBufObj sciBufObj, uint32_t &packetId)
{
    std::vector<NvSciBufObj>::iterator it = std::find_if(
        bufObjs.begin(), bufObjs.end(), [sciBufObj](const NvSciBufObj &obj) { return (sciBufObj == obj); });

    if (bufObjs.end() == it) {
        // Didn't find matching buffer
        PLOG_ERR("MapPayload, failed to get packet index for buffer\n");
        return NVSIPL_STATUS_ERROR;
    }

    packetId = std::distance(bufObjs.begin(), it);

    return NVSIPL_STATUS_OK;
}

SIPLStatus CVICProducer::MapPayload( void *pBuffer, uint32_t &packetIndex )
{
    CBuffer* pBuf = reinterpret_cast<CBuffer*>( pBuffer );

    NvSciBufObj sciBufObj = pBuf->GetNvSciBufObj();
    PCHK_PTR_AND_RETURN( sciBufObj, "CBuffer::GetNvSciBufObj" );

    uint32_t i = 0;
    for ( ; i < MAX_NUM_PACKETS; i++ )
    {
        if ( sciBufObj == m_vBufObjs[i] )
        {
            break;
        }
    }

    if ( i == MAX_NUM_PACKETS )
    {
        // Didn't find matching buffer
        PLOG_ERR( "MapPayload, failed to get packet index for buffer\n" );
        return NVSIPL_STATUS_ERROR;
    }

    packetIndex = i;
    m_pBuffers[packetIndex] = pBuf;
    //m_pBuffers[packetIndex]->AddRef();
    return NVSIPL_STATUS_OK;
}
SIPLStatus CVICProducer::Post(void *pBuffer)
{
    uint32_t packetIndex = 0;
    auto status = NVSIPL_STATUS_OK;
    auto sciErr = NvSciError_Success;

    const INvSIPLClient::ConsumerDesc::OutputType &outputType = INvSIPLClient::ConsumerDesc::OutputType::ISP0;
    status = MapPayload(pBuffer, packetIndex);
    PCHK_STATUS_AND_RETURN(status, "MapPayload");

    NvSciSyncFence postFence = NvSciSyncFenceInitializer;
    status = GetPostfence(outputType, packetIndex, &postFence);
    PCHK_STATUS_AND_RETURN(status, "GetPostFence");

    uint32_t elementId = 0U;
    status = GetElemIdByUserType(ELEMENT_TYPE_NV12_BL, elementId);
    /* status = GetElemIdByUserType(m_outputTypeToElemType[static_cast<uint32_t>(outputType)], elementId); */
    PCHK_STATUS_AND_RETURN(status, "GetElemIdByUserType");

    /* Update postfence for this element */
    sciErr = NvSciStreamBlockPacketFenceSet(m_handle, m_packets[packetIndex].handle, elementId, &postFence);
    NvSciSyncFenceClear(&postFence);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockPacketFenceSet");


    sciErr = NvSciStreamProducerPacketPresent(m_handle, m_packets[packetIndex].handle);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamProducerPacketPresent");

    m_numBuffersWithConsumer++;
    PLOG_DBG("Post, m_numBuffersWithConsumer: %u\n", m_numBuffersWithConsumer.load());

    if (m_pProfiler != nullptr) {
        m_pProfiler->OnFrameAvailable();
    }

    return NVSIPL_STATUS_OK;
}
