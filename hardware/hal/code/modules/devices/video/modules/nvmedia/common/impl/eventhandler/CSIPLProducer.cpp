// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include <algorithm>

#include "CSIPLProducer.hpp"
#include "CPoolManager.hpp"

constexpr static int32_t OUTPUT_TYPE_UNDEFINED = -1;

CSIPLProducer::CSIPLProducer(NvSciStreamBlock handle, uint32_t uSensor, uint32_t i_outputtype, INvSIPLCamera* pCamera, std::shared_ptr<CAttributeProvider> attrProvider) :
    CProducer("CSIPLProducer", handle, uSensor, attrProvider)
{
    m_pCamera = pCamera;

    memset(m_elemTypeToOutputType, OUTPUT_TYPE_UNDEFINED, sizeof(m_elemTypeToOutputType));
    memset(m_outputTypeToElemType, ELEMENT_TYPE_UNDEFINED, sizeof(m_outputTypeToElemType));
    m_outputType = (INvSIPLClient::ConsumerDesc::OutputType)i_outputtype;
}

CSIPLProducer::~CSIPLProducer(void)
{
    PLOG_DBG("Release.\n");

    for (uint32_t i = 0U; i < MAX_OUTPUTS_PER_SENSOR; ++i) {
        for (uint32_t j = 0U; j < m_siplBuffers[i].bufObjs.size(); ++j) {
            if (m_siplBuffers[i].bufObjs[j] != nullptr) {
                NvSciBufObjFree(m_siplBuffers[i].bufObjs[j]);
                m_siplBuffers[i].bufObjs[j] = nullptr;
        }
    }
        }
    }

SIPLStatus CSIPLProducer::MapElemTypeToOutputType(PacketElementType userType,
                                                  INvSIPLClient::ConsumerDesc::OutputType &outputType)
{
    if (m_elemTypeToOutputType[userType] != OUTPUT_TYPE_UNDEFINED) {
        outputType = static_cast<INvSIPLClient::ConsumerDesc::OutputType>(m_elemTypeToOutputType[userType]);
    } else {
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CSIPLProducer::HandleClientInit(void)
{
    m_elemTypeToOutputType[ELEMENT_TYPE_NV12_BL] = static_cast<int32_t>(INvSIPLClient::ConsumerDesc::OutputType::ISP0);
    m_elemTypeToOutputType[ELEMENT_TYPE_NV12_PL] = static_cast<int32_t>(INvSIPLClient::ConsumerDesc::OutputType::ISP1);
    m_elemTypeToOutputType[ELEMENT_TYPE_ICP_RAW] = static_cast<int32_t>(INvSIPLClient::ConsumerDesc::OutputType::ICP);

    // create raw buf attrlist
    /* auto sciErr = NvSciBufAttrListCreate(m_sciBufModule, &m_rawBufAttrList); */
    /* PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListCreate"); */

    /* auto status = SetBufAttrList(INvSIPLClient::ConsumerDesc::OutputType::ICP, m_rawBufAttrList); */
    /* PCHK_STATUS_AND_RETURN(status, "SetBufAttrList for RAW"); */

    return NVSIPL_STATUS_OK;
}

// Create and set CPU signaler and waiter attribute lists.
SIPLStatus CSIPLProducer::SetSyncAttrList(PacketElementType userType,
                                          NvSciSyncAttrList &signalerAttrList,
                                          NvSciSyncAttrList &waiterAttrList)
{
    INvSIPLClient::ConsumerDesc::OutputType outputType;  
    auto status = MapElemTypeToOutputType(userType, outputType);
    PLOG_DBG("%s:SetSyncAttrList userType = %d,outputType=%d\n",m_name.c_str(),userType,outputType);
    PCHK_STATUS_AND_RETURN(status, "MapElemTypeToOutputType");

    status = m_pCamera->FillNvSciSyncAttrList(m_uSensorId, outputType, signalerAttrList, SIPL_SIGNALER);
    PCHK_STATUS_AND_RETURN(status, "Signaler INvSIPLCamera::FillNvSciSyncAttrList");

#ifdef NVMEDIA_QNX
    status = m_pCamera->FillNvSciSyncAttrList(m_uSensorId, outputType, waiterAttrList, SIPL_WAITER);
    PCHK_STATUS_AND_RETURN(status, "Waiter INvSIPLCamera::FillNvSciSyncAttrList");
#else
    NvSciSyncAttrKeyValuePair keyValue[2];
    bool cpuWaiter = true;
    keyValue[0].attrKey = NvSciSyncAttrKey_NeedCpuAccess;
    keyValue[0].value = (void *)&cpuWaiter;
    keyValue[0].len = sizeof(cpuWaiter);
    NvSciSyncAccessPerm cpuPerm = NvSciSyncAccessPerm_WaitOnly;
    keyValue[1].attrKey = NvSciSyncAttrKey_RequiredPerm;
    keyValue[1].value = (void *)&cpuPerm;
    keyValue[1].len = sizeof(cpuPerm);
    auto sciErr = NvSciSyncAttrListSetAttrs(waiterAttrList, keyValue, 2);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "CPU waiter NvSciSyncAttrListSetAttrs");
#endif // NVMEDIA_QNX

    return NVSIPL_STATUS_OK;
}

SIPLStatus CSIPLProducer::SetDataBufAttrList(PacketElementType userType, NvSciBufAttrList &bufAttrList)
{
    INvSIPLClient::ConsumerDesc::OutputType outputType;
    auto status = MapElemTypeToOutputType(userType, outputType);
    PLOG_DBG("%s:SetDataBufAttrList userType = %d;status=%d\n",m_name.c_str(),userType,status);
    PCHK_STATUS_AND_RETURN(status, "MapElemTypeToOutputType");

    status = SetBufAttrList(userType, outputType, bufAttrList);
    PCHK_STATUS_AND_RETURN(status, "SetBufAttrList");

    return NVSIPL_STATUS_OK;
}

// Buffer setup functions
SIPLStatus CSIPLProducer::SetBufAttrList(PacketElementType userType,
                                         INvSIPLClient::ConsumerDesc::OutputType outputType,
                                         NvSciBufAttrList &bufAttrList)
{
    NvSciBufAttrValAccessPerm access_perm = NvSciBufAccessPerm_ReadWrite;
    NvSciBufAttrKeyValuePair attrKvp = {NvSciBufGeneralAttrKey_RequiredPerm,
                                        &access_perm,
                                        sizeof(access_perm)};
    auto sciErr = NvSciBufAttrListSetAttrs(bufAttrList, &attrKvp, 1);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");

    if (outputType != INvSIPLClient::ConsumerDesc::OutputType::ICP) {
        bool isCpuAcccessReq = true;
        bool isCpuCacheEnabled = true;

        NvSciBufAttrKeyValuePair setAttrs[] = {
            { NvSciBufGeneralAttrKey_NeedCpuAccess, &isCpuAcccessReq, sizeof(isCpuAcccessReq) },
            { NvSciBufGeneralAttrKey_EnableCpuCache, &isCpuCacheEnabled, sizeof(isCpuCacheEnabled) }
        };
        sciErr = NvSciBufAttrListSetAttrs(bufAttrList, setAttrs, 2U);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");

        NvSciBufAttrValImageLayoutType layout = NvSciBufImage_PitchLinearType;
        if (ELEMENT_TYPE_NV12_BL == userType) {
            layout = NvSciBufImage_BlockLinearType;
        } else if (ELEMENT_TYPE_NV12_PL == userType) {
            layout = NvSciBufImage_PitchLinearType;
        } else {
            LOG_ERR("SetDataBufAttrList: Unsuported ISP output type. \n");
            return NVSIPL_STATUS_ERROR;
        }

        NvSciBufSurfSampleType surfSampleType = NvSciSurfSampleType_420;
        NvSciBufSurfBPC surfBPC = NvSciSurfBPC_8;
        NvSciBufType bufType = NvSciBufType_Image;
        NvSciBufSurfType surfType = NvSciSurfType_YUV;
        NvSciBufSurfMemLayout surfMemLayout = NvSciSurfMemLayout_SemiPlanar;
        NvSciBufSurfComponentOrder surfCompOrder = NvSciSurfComponentOrder_YUV;
        NvSciBufAttrValColorStd surfColorStd[] = { NvSciColorStd_REC709_ER };
        NvSciBufAttrKeyValuePair keyVals[] = {
            { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
            { NvSciBufImageAttrKey_SurfType, &surfType, sizeof(surfType) },
            { NvSciBufImageAttrKey_SurfBPC, &surfBPC, sizeof(surfBPC) },
            { NvSciBufImageAttrKey_SurfMemLayout, &surfMemLayout, sizeof(surfMemLayout) },
            { NvSciBufImageAttrKey_SurfSampleType, &surfSampleType, sizeof(surfSampleType) },
            { NvSciBufImageAttrKey_SurfComponentOrder, &surfCompOrder, sizeof(surfCompOrder) },
            { NvSciBufImageAttrKey_SurfColorStd, &surfColorStd, sizeof(surfColorStd) },
            { NvSciBufImageAttrKey_Layout, &layout, sizeof(layout) }
        };

        size_t length = sizeof(keyVals) / sizeof(NvSciBufAttrKeyValuePair);
        auto err = NvSciBufAttrListSetAttrs(bufAttrList, keyVals, length);
        CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufAttrListSetAttrs");
    }

    auto status = m_pCamera->GetImageAttributes(m_uSensorId, outputType, bufAttrList);
    PCHK_STATUS_AND_RETURN(status, "GetImageAttributes");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CSIPLProducer::MapDataBuffer(PacketElementType userType, uint32_t packetIndex, NvSciBufObj bufObj)
{
    PLOG_DBG("%s:Mapping data buffer, userType: %u, packetIndex: %u.\n", m_name.c_str(), userType, packetIndex);

    NvSciBufObj dupObj;
    auto sciErr = NvSciBufObjDup(bufObj, &dupObj);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjDup");

    INvSIPLClient::ConsumerDesc::OutputType outputType;
    auto status = MapElemTypeToOutputType(userType, outputType);
    PCHK_STATUS_AND_RETURN(status, "MapElemTypeToOutputType");

    m_outputTypeToElemType[static_cast<uint32_t>(outputType)] = userType;

    m_siplBuffers[static_cast<uint32_t>(outputType)].bufObjs.push_back(std::move(dupObj));

    return NVSIPL_STATUS_OK;
}

// Create client buffer objects from NvSciBufObj
SIPLStatus CSIPLProducer::MapMetaBuffer(uint32_t packetIndex, NvSciBufObj bufObj)
{
    PLOG_DBG("%s:Mapping meta buffer, packetIndex: %u.\r\n", m_name.c_str(), packetIndex);
    auto sciErr = NvSciBufObjGetCpuPtr(bufObj, (void**)&m_metaPtrs[packetIndex]);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjGetCpuPtr");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CSIPLProducer::RegisterSignalSyncObj(PacketElementType userType, NvSciSyncObj signalSyncObj)
{
    INvSIPLClient::ConsumerDesc::OutputType outputType;
    auto status = MapElemTypeToOutputType(userType, outputType);
    PCHK_STATUS_AND_RETURN(status, "MapElemTypeToOutputType");

    //For ISP sync, only one signalSyncObj.
    status = m_pCamera->RegisterNvSciSyncObj(m_uSensorId, outputType, NVSIPL_EOFSYNCOBJ, signalSyncObj);
    PCHK_STATUS_AND_RETURN(status, "INvSIPLCamera::RegisterNvSciSyncObj");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CSIPLProducer::RegisterWaiterSyncObj(PacketElementType userType, NvSciSyncObj waiterSyncObj)
{
#ifdef NVMEDIA_QNX
    INvSIPLClient::ConsumerDesc::OutputType outputType;
    auto status = MapElemTypeToOutputType(userType, outputType);
    PCHK_STATUS_AND_RETURN(status, "MapElemTypeToOutputType");

    status = m_pCamera->RegisterNvSciSyncObj(m_uSensorId, outputType, NVSIPL_PRESYNCOBJ, waiterSyncObj);

    PCHK_STATUS_AND_RETURN(status, "INvSIPLCamera::RegisterNvSciSyncObj");
#endif // NVMEDIA_QNX

    return NVSIPL_STATUS_OK;
}

SIPLStatus CSIPLProducer::HandleSetupComplete(void)
{
    auto status = CProducer::HandleSetupComplete();
    PCHK_STATUS_AND_RETURN(status, "HandleSetupComplete");

    status = RegisterBuffers();
    PCHK_STATUS_AND_RETURN(status, "RegisterBuffers");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CSIPLProducer::RegisterBuffers(void)
{
    PLOG_DBG("%s:RegisterBuffers\r\n", m_name.c_str());
    
    for (uint32_t i = 0U; i < MAX_OUTPUTS_PER_SENSOR; ++i) {
        if (!m_siplBuffers[i].bufObjs.empty()) {
            auto status = m_pCamera->RegisterImages(
                m_uSensorId, static_cast<INvSIPLClient::ConsumerDesc::OutputType>(i), m_siplBuffers[i].bufObjs);
            PCHK_STATUS_AND_RETURN(status, "INvSIPLCamera::RegisterImages");

            m_siplBuffers[i].nvmBuffers.resize(m_siplBuffers[i].bufObjs.size(), nullptr);
        }
    }

    return NVSIPL_STATUS_OK;
}

//Before calling PreSync, m_nvmBuffers[packetIndex] should already be filled.
SIPLStatus CSIPLProducer::InsertPrefence(PacketElementType userType, uint32_t packetIndex, NvSciSyncFence &prefence)
{
#ifdef NVMEDIA_QNX
    PLOG_DBG("AddPrefence, packetIndex: %u\n", packetIndex);

    INvSIPLClient::ConsumerDesc::OutputType outputType;
    auto status = MapElemTypeToOutputType(userType, outputType);
    PCHK_STATUS_AND_RETURN(status, "MapElemTypeToOutputType");

    status = m_siplBuffers[static_cast<uint32_t>(outputType)].nvmBuffers[packetIndex]->AddNvSciSyncPrefence(prefence);

    PCHK_STATUS_AND_RETURN(status, "AddNvSciSyncPrefence");
#endif

    return NVSIPL_STATUS_OK;
}

SIPLStatus CSIPLProducer::GetPostfence(INvSIPLClient::ConsumerDesc::OutputType outputType,
                                       uint32_t packetIndex,
                                       NvSciSyncFence *pPostfence)
{
    auto status =
        m_siplBuffers[static_cast<uint32_t>(outputType)].nvmBuffers[packetIndex]->GetEOFNvSciSyncFence(pPostfence);
    PCHK_STATUS_AND_RETURN(status, "GetEOFNvSciSyncFence");

    return NVSIPL_STATUS_OK;
}

void CSIPLProducer::OnPacketGotten(uint32_t packetIndex)
{
    for (uint32_t i = 0U; i < MAX_OUTPUTS_PER_SENSOR; ++i) {
        if (!m_siplBuffers[i].nvmBuffers.empty() && m_siplBuffers[i].nvmBuffers[packetIndex] != nullptr) {
            m_siplBuffers[i].nvmBuffers[packetIndex]->Release();
            LOG_DBG("CSIPLProducer::OnPacketGotten, ISP type %u packet gotten \n", i);
        }
    }
}

SIPLStatus CSIPLProducer::GetPacketId(std::vector<NvSciBufObj> bufObjs, NvSciBufObj sciBufObj, uint32_t &packetId)
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

SIPLStatus
CSIPLProducer::MapPayload(INvSIPLClient::ConsumerDesc::OutputType outputType, void *pBuffer, uint32_t &packetIndex)
{
    INvSIPLClient::INvSIPLNvMBuffer *pNvMBuf = reinterpret_cast<INvSIPLClient::INvSIPLNvMBuffer *>(pBuffer);
    NvSciBufObj sciBufObj = pNvMBuf->GetNvSciBufImage();
    PCHK_PTR_AND_RETURN(sciBufObj, "INvSIPLClient::INvSIPLNvMBuffer::GetNvSciBufImage");

    const uint32_t &siplBufId = static_cast<uint32_t>(outputType);
    auto status = GetPacketId(m_siplBuffers[siplBufId].bufObjs, sciBufObj, packetIndex);
    PCHK_STATUS_AND_RETURN(status, "ISP GetPacketId");

    pNvMBuf->AddRef();
    m_siplBuffers[siplBufId].nvmBuffers[packetIndex] = pNvMBuf;

    if (m_metaPtrs[packetIndex] != nullptr) {
        const INvSIPLClient::ImageMetaData &md = pNvMBuf->GetImageData();
        static_cast<MetaData *>(m_metaPtrs[packetIndex])->frameCaptureTSC = md.frameCaptureTSC;
        static_cast<MetaData *>(m_metaPtrs[packetIndex])->frameCaptureStartTSC = md.frameCaptureStartTSC;
        memcpy(static_cast<MetaData *>(m_metaPtrs[packetIndex])->exposureTime, md.sensorExpInfo.exposureTime,
            sizeof(float_t)*DEVBLK_CDI_MAX_EXPOSURES);
#if 0
        printf("CSIPLProducer outputType:%d m_uSensorId:%d frameCaptureTSC:%ld frameCaptureStartTSC:%ld\n", outputType, m_uSensorId, md.frameCaptureTSC, md.frameCaptureStartTSC);
        printf("CSIPLProducer expTimeValid:%d exposureTime[%f][%f][%f][%f][%f][%f][%f][%f]\n", md.sensorExpInfo.expTimeValid, md.sensorExpInfo.exposureTime[0],
            md.sensorExpInfo.exposureTime[1], md.sensorExpInfo.exposureTime[2], md.sensorExpInfo.exposureTime[3], md.sensorExpInfo.exposureTime[4],
            md.sensorExpInfo.exposureTime[5], md.sensorExpInfo.exposureTime[6], md.sensorExpInfo.exposureTime[7]);
#endif
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CSIPLProducer::Post(void *pBuffer)
{
    uint32_t packetIndex = 0;
    auto status = NVSIPL_STATUS_OK;
    auto sciErr = NvSciError_Success;

    NvSIPLBuffers &siplBuffers = *(static_cast<NvSIPLBuffers *>(pBuffer));
    for (uint32_t i = 0U; i < siplBuffers.size(); ++i) {
        const INvSIPLClient::ConsumerDesc::OutputType &outputType = siplBuffers[i].first;
        status = MapPayload(outputType, siplBuffers[i].second, packetIndex);
        PCHK_STATUS_AND_RETURN(status, "MapPayload");

        NvSciSyncFence postFence = NvSciSyncFenceInitializer;
        status = GetPostfence(outputType, packetIndex, &postFence);
        PCHK_STATUS_AND_RETURN(status, "GetPostFence");

        uint32_t elementId = 0U;
        status = GetElemIdByUserType(m_outputTypeToElemType[static_cast<uint32_t>(outputType)], elementId);
        PCHK_STATUS_AND_RETURN(status, "GetElemIdByUserType");

        /* Update postfence for this element */
        sciErr = NvSciStreamBlockPacketFenceSet(m_handle, m_packets[packetIndex].handle, elementId, &postFence);
        NvSciSyncFenceClear(&postFence);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockPacketFenceSet");
    }

    sciErr = NvSciStreamProducerPacketPresent(m_handle, m_packets[packetIndex].handle);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamProducerPacketPresent");

    m_numBuffersWithConsumer++;
    PLOG_DBG("Post, m_numBuffersWithConsumer: %u\n", m_numBuffersWithConsumer.load());

    if (m_pProfiler != nullptr) {
        m_pProfiler->OnFrameAvailable();
    }

    return NVSIPL_STATUS_OK;
}
