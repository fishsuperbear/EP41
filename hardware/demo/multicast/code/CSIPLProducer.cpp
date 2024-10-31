// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include "CSIPLProducer.hpp"
#include "CPoolManager.hpp"

CSIPLProducer::CSIPLProducer(NvSciStreamBlock handle, uint32_t uSensor, INvSIPLCamera* pCamera) :
    CProducer("CSIPLProducer", handle, uSensor)
{
    m_pCamera = pCamera;

    m_vRawBufObjs.resize(MAX_NUM_PACKETS);
    m_vIspBufObjs.resize(MAX_NUM_PACKETS);

    for (uint32_t i = 0; i < MAX_NUM_PACKETS; i++) {
        m_vRawBufObjs[i] = nullptr;
        m_vIspBufObjs[i] = nullptr;
    }
    m_ispOutputType = INvSIPLClient::ConsumerDesc::OutputType::ISP0;
}

CSIPLProducer::~CSIPLProducer(void)
{
    PLOG_DBG("Release.\n");
    for (NvSciBufObj bufObj : m_vRawBufObjs) {
        if (bufObj != nullptr) {
            NvSciBufObjFree(bufObj);
        }
    }
    std::vector<NvSciBufObj>().swap(m_vRawBufObjs);

    for (NvSciBufObj bufObj : m_vIspBufObjs) {
        if (bufObj != nullptr) {
            NvSciBufObjFree(bufObj);
        }
    }
    std::vector<NvSciBufObj>().swap(m_vIspBufObjs);

    if (m_rawBufAttrList != nullptr) {
        NvSciBufAttrListFree(m_rawBufAttrList);
        m_rawBufAttrList = nullptr;
    }
}

SIPLStatus CSIPLProducer::HandleClientInit(void)
{
    return NVSIPL_STATUS_OK;
}

// Create and set CPU signaler and waiter attribute lists.
SIPLStatus CSIPLProducer::SetSyncAttrList(void)
{
    auto status = m_pCamera->FillNvSciSyncAttrList(m_uSensorId,
                                                   m_ispOutputType,
                                                   m_signalerAttrList,
                                                   SIPL_SIGNALER);
    PCHK_STATUS_AND_RETURN(status, "Signaler INvSIPLCamera::FillNvSciSyncAttrList");

#ifdef NVMEDIA_QNX
    status = m_pCamera->FillNvSciSyncAttrList(m_uSensorId,
                                              m_ispOutputType,
                                              m_waiterAttrList,
                                              SIPL_WAITER);
    PCHK_STATUS_AND_RETURN(status, "Waiter INvSIPLCamera::FillNvSciSyncAttrList");
#else
    NvSciSyncAttrKeyValuePair keyValue[2];
    bool cpuWaiter = true;
    keyValue[0].attrKey = NvSciSyncAttrKey_NeedCpuAccess;
    keyValue[0].value = (void *)&cpuWaiter;
    keyValue[0].len = sizeof(cpuWaiter);
    NvSciSyncAccessPerm cpuPerm = NvSciSyncAccessPerm_WaitOnly;
    keyValue[1].attrKey = NvSciSyncAttrKey_RequiredPerm;
    keyValue[1].value = (void*)&cpuPerm;
    keyValue[1].len = sizeof(cpuPerm);
    auto sciErr = NvSciSyncAttrListSetAttrs(m_waiterAttrList, keyValue, 2);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "CPU waiter NvSciSyncAttrListSetAttrs");
#endif // NVMEDIA_QNX

    return NVSIPL_STATUS_OK;
}

SIPLStatus CSIPLProducer::CreateBufAttrLists(NvSciBufModule bufModule)
{
    //Create ISP buf attrlist
    auto status = CClientCommon::CreateBufAttrLists(bufModule);
    CHK_STATUS_AND_RETURN(status, "CClientCommon::CreateBufAttrList");

    // create raw buf attrlist
    auto sciErr = NvSciBufAttrListCreate(bufModule, &m_rawBufAttrList);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListCreate");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CSIPLProducer::SetDataBufAttrList(void)
{
    auto status = SetBufAttrList(INvSIPLClient::ConsumerDesc::OutputType::ICP, m_rawBufAttrList);
    PCHK_STATUS_AND_RETURN(status, "SetBufAttrList for RAW");

    status = SetBufAttrList(m_ispOutputType, m_bufAttrLists[DATA_ELEMENT_INDEX]);
    PCHK_STATUS_AND_RETURN(status, "SetBufAttrList for ISP");

    return NVSIPL_STATUS_OK;
}

// Buffer setup functions
SIPLStatus CSIPLProducer::SetBufAttrList(
    INvSIPLClient::ConsumerDesc::OutputType outputType,
    NvSciBufAttrList& bufAttrList)
{
    NvSciBufAttrValAccessPerm access_perm = NvSciBufAccessPerm_ReadWrite;
    NvSciBufAttrKeyValuePair attrKvp = {NvSciBufGeneralAttrKey_RequiredPerm,
                                        &access_perm,
                                        sizeof(access_perm)};
    auto sciErr = NvSciBufAttrListSetAttrs(bufAttrList, &attrKvp, 1);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");

    if (outputType == INvSIPLClient::ConsumerDesc::OutputType::ISP0) {
        bool isCpuAcccessReq = true;
        bool isCpuCacheEnabled = true;

        NvSciBufAttrKeyValuePair setAttrs[] = {
            { NvSciBufGeneralAttrKey_NeedCpuAccess, &isCpuAcccessReq, sizeof(isCpuAcccessReq) },
            { NvSciBufGeneralAttrKey_EnableCpuCache, &isCpuCacheEnabled, sizeof(isCpuCacheEnabled) },
        };
        sciErr = NvSciBufAttrListSetAttrs(bufAttrList, setAttrs, 2);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");
    }

    auto status = m_pCamera->GetImageAttributes(m_uSensorId, outputType, bufAttrList);
    PCHK_STATUS_AND_RETURN(status, "GetImageAttributes");

    return NVSIPL_STATUS_OK;
}

// Create client buffer objects from NvSciBufObj
SIPLStatus CSIPLProducer::MapDataBuffer(uint32_t packetIndex)
{
    PLOG_DBG("Mapping data buffer, packetIndex: %u.\n", packetIndex);
    auto sciErr = NvSciBufObjDup(m_packets[packetIndex].dataObj, &m_vIspBufObjs[packetIndex]);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjDup");

    return NVSIPL_STATUS_OK;
}

// Create client buffer objects from NvSciBufObj
SIPLStatus CSIPLProducer::MapMetaBuffer(uint32_t packetIndex)
{
    PLOG_DBG("Mapping meta buffer, packetIndex: %u.\n", packetIndex);
    auto sciErr = NvSciBufObjGetCpuPtr(m_packets[packetIndex].metaObj, (void**)&m_metaPtrs[packetIndex]);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjGetCpuPtr");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CSIPLProducer::RegisterSignalSyncObj(void)
{
    //Only one signalSyncObj
    auto status = m_pCamera->RegisterNvSciSyncObj(m_uSensorId, m_ispOutputType, NVSIPL_EOFSYNCOBJ, m_signalSyncObj);
    PCHK_STATUS_AND_RETURN(status, "INvSIPLCamera::RegisterNvSciSyncObj");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CSIPLProducer::RegisterWaiterSyncObj(uint32_t index)
{
#ifdef NVMEDIA_QNX
    auto status = m_pCamera->RegisterNvSciSyncObj(m_uSensorId,
                                                  m_ispOutputType,
                                                  NVSIPL_PRESYNCOBJ,
                                                  m_waiterSyncObjs[index]);
    PCHK_STATUS_AND_RETURN(status, "INvSIPLCamera::RegisterNvSciSyncObj");
#endif // NVMEDIA_QNX

    return NVSIPL_STATUS_OK;
}

SIPLStatus CSIPLProducer::HandleSetupComplete(void)
{
    auto status = CProducer::HandleSetupComplete();
    PCHK_STATUS_AND_RETURN(status, "HandleSetupComplete");

    //Alloc raw buffers
    status = CPoolManager::ReconcileAndAllocBuffers(m_rawBufAttrList, m_vRawBufObjs);
    PCHK_STATUS_AND_RETURN(status, "CPoolManager::ReconcileAndAllocBuffers");

    status = RegisterBuffers();
    PCHK_STATUS_AND_RETURN(status, "RegisterBuffers");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CSIPLProducer::RegisterBuffers(void)
{
    PLOG_DBG("RegisterBuffers\n");
    auto status = m_pCamera->RegisterImages(m_uSensorId, INvSIPLClient::ConsumerDesc::OutputType::ICP, m_vRawBufObjs);
    PCHK_STATUS_AND_RETURN(status, "INvSIPLCamera::RegisterImages");

    status = m_pCamera->RegisterImages(m_uSensorId, m_ispOutputType, m_vIspBufObjs);
    PCHK_STATUS_AND_RETURN(status, "INvSIPLCamera::RegisterImages");

    return NVSIPL_STATUS_OK;
}

//Before calling PreSync, m_nvmBuffers[packetIndex] should already be filled.
SIPLStatus CSIPLProducer::InsertPrefence(uint32_t packetIndex, NvSciSyncFence &prefence)
{
#ifdef NVMEDIA_QNX
    PLOG_DBG("AddPrefence, packetIndex: %u\n", packetIndex);
    auto status = m_nvmBuffers[packetIndex]->AddNvSciSyncPrefence(prefence);
    PCHK_STATUS_AND_RETURN(status, "AddNvSciSyncPrefence");
#endif // NVMEDIA_QNX

    return NVSIPL_STATUS_OK;
}

SIPLStatus CSIPLProducer::GetPostfence(uint32_t packetIndex, NvSciSyncFence *pPostfence)
{
    auto status = m_nvmBuffers[packetIndex]->GetEOFNvSciSyncFence(pPostfence);
    PCHK_STATUS_AND_RETURN(status, "GetEOFNvSciSyncFence");

    return NVSIPL_STATUS_OK;
}

void CSIPLProducer::OnPacketGotten(uint32_t packetIndex)
{
    m_nvmBuffers[packetIndex]->Release();
}

SIPLStatus CSIPLProducer::MapPayload(void *pBuffer, uint32_t& packetIndex)
{
    INvSIPLClient::INvSIPLNvMBuffer* pNvMBuf = reinterpret_cast<INvSIPLClient::INvSIPLNvMBuffer*>(pBuffer);

    NvSciBufObj sciBufObj = pNvMBuf->GetNvSciBufImage();
    PCHK_PTR_AND_RETURN(sciBufObj, "INvSIPLClient::INvSIPLNvMBuffer::GetNvSciBufImage");
    uint32_t i = 0;
    for (; i < MAX_NUM_PACKETS; i++) {
        if (sciBufObj == m_vIspBufObjs[i]) {
            break;
        }
    }
    if (i == MAX_NUM_PACKETS) {
        // Didn't find matching buffer
        PLOG_ERR("MapPayload, failed to get packet index for buffer\n");
        return NVSIPL_STATUS_ERROR;
    }
    packetIndex = i;
    if (m_metaPtrs[packetIndex] != nullptr) {
        const INvSIPLClient::ImageMetaData &md = pNvMBuf->GetImageData();
        m_metaPtrs[packetIndex]->frameCaptureTSC = md.frameCaptureTSC;
    }
    m_nvmBuffers[packetIndex] = pNvMBuf;
    m_nvmBuffers[packetIndex]->AddRef();

    return NVSIPL_STATUS_OK;
}
