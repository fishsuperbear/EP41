// Copyright (c) 2022-2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include "CStitchingConsumer.hpp"
#include "nvscibuf.h"

CStitchingConsumer::CStitchingConsumer(NvSciStreamBlock handle, uint32_t uSensorId, NvSciStreamBlock queueHandle)
    : CConsumer("CStitchingConsumer", handle, uSensorId, queueHandle)
{
}

void CStitchingConsumer::PreInit(std::shared_ptr<CDisplayProducer> &spDisplayProd)
{
    m_spDisplayProd = spDisplayProd;

    for (uint32_t i = 0U; i < MAX_NUM_PACKETS; ++i) {
        m_srcBufObjs[i] = nullptr;
    }
}

SIPLStatus CStitchingConsumer::HandleClientInit()
{
    NvMedia2D *p2dHandle = nullptr;
    NvMediaStatus nvmStatus = NvMedia2DCreate(&p2dHandle, nullptr);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DCreate");

    m_up2DDevice.reset(p2dHandle);

    m_spDisplayProd->RegisterCompositor(m_uSensorId, this);

    return NVSIPL_STATUS_OK;
}

CStitchingConsumer::~CStitchingConsumer(void)
{
    LOG_DBG("CStitchingConsumer released.\n");
}

SIPLStatus CStitchingConsumer::Deinit(void)
{
    LOG_DBG("CStitchingConsumer::Deinit start\n");

    auto nvmStatus = NvMedia2DUnregisterNvSciSyncObj(m_up2DDevice.get(), m_2DSignalSyncObj);
    if (nvmStatus != NVMEDIA_STATUS_OK) {
        LOG_ERR("CStitchingConsumer::NvMedia2DUnregisterNvSciSyncObj signal sync obj status = %u\n", nvmStatus);
    }

    for (uint32_t i = 0U; i < m_numWaitSyncObj; ++i) {
        for (uint32_t j = 0U; j < m_elemsInfo.size(); ++j) {
            if (nullptr == m_waiterSyncObjs[i][j]) {
                continue;
            }

            nvmStatus = NvMedia2DUnregisterNvSciSyncObj(m_up2DDevice.get(), m_waiterSyncObjs[i][j]);
            if (nvmStatus != NVMEDIA_STATUS_OK) {
                LOG_ERR("CStitchingConsumer::NvMedia2DUnregisterNvSciSyncObj waiter sync obj status = %u\n", nvmStatus);
            }
        }
    }

    for (NvSciBufObj bufObj : m_dstBufObjs) {
        auto status = UnRegisterDstNvSciObjBuf(bufObj);
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("CStitchingConsumer::UnRegisterDstNvSciObjBuf dst buf object status = %u\n", status);
        }
    }

    for (uint32_t i = 0; i < MAX_NUM_PACKETS; ++i) {
        if (m_srcBufObjs[i] != nullptr) {
            auto status = UnRegisterDstNvSciObjBuf(m_srcBufObjs[i]);
            if (status != NVSIPL_STATUS_OK) {
                LOG_ERR("CStitchingConsumer::UnRegisterDstNvSciObjBuf src buf object status = %u\n", status);
            }
        }
    }

    LOG_DBG("CStitchingConsumer::Deinit end\n");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CStitchingConsumer::RegisterDstNvSciObjBuf(NvSciBufObj obj)
{
    NvMediaStatus nvmStatus = NvMedia2DRegisterNvSciBufObj(m_up2DDevice.get(), obj);
    CHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DRegisterNvSciBufObj");

    m_dstBufObjs.push_back(obj);
    return NVSIPL_STATUS_OK;
}

SIPLStatus CStitchingConsumer::UnRegisterDstNvSciObjBuf(NvSciBufObj obj)
{
    NvMediaStatus nvmStatus = NvMedia2DUnregisterNvSciBufObj(m_up2DDevice.get(), obj);
    CHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DUnregisterNvSciBufObj");

    return NVSIPL_STATUS_OK;
}

// Buffer setup functions
SIPLStatus CStitchingConsumer::SetDataBufAttrList(NvSciBufAttrList &bufAttrList)
{
    NvSciBufAttrValAccessPerm accessPerm = NvSciBufAccessPerm_ReadWrite;
    NvSciBufType bufType = NvSciBufType_Image;

    /* Set all key-value pairs */
    NvSciBufAttrKeyValuePair attributes[] = {
        { NvSciBufGeneralAttrKey_RequiredPerm, &accessPerm, sizeof(accessPerm) },
        { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
    };

    auto sciErr =
        NvSciBufAttrListSetAttrs(bufAttrList, attributes, sizeof(attributes) / sizeof(NvSciBufAttrKeyValuePair));
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");

    auto status = NvMedia2DFillNvSciBufAttrList(m_up2DDevice.get(), bufAttrList);
    PCHK_NVMSTATUS_AND_RETURN(status, "NvMedia2DFillNvSciBufAttrList");

    sciErr = NvSciBufAttrListClone(bufAttrList, &m_dataBufAttrList);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListClone");

    return NVSIPL_STATUS_OK;
}

// Sync object setup functions
SIPLStatus CStitchingConsumer::SetSyncAttrList(NvSciSyncAttrList &signalerAttrList, NvSciSyncAttrList &waiterAttrList)
{
    NvMediaStatus nvmStatus = NvMedia2DFillNvSciSyncAttrList(m_up2DDevice.get(), signalerAttrList, NVMEDIA_SIGNALER);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DFillNvSciSyncAttrList");

    nvmStatus = NvMedia2DFillNvSciSyncAttrList(m_up2DDevice.get(), waiterAttrList, NVMEDIA_WAITER);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DFillNvSciSyncAttrList");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CStitchingConsumer::MapDataBuffer(uint32_t packetIndex, NvSciBufObj bufObj)
{
    PLOG_DBG("Mapping data buffer, packetIndex: %u.\n", packetIndex);

    m_srcBufObjs[packetIndex] = bufObj;
    PCHK_PTR_AND_RETURN(m_up2DDevice.get(), "NvMedia2DDevice nullptr");
    NvMediaStatus nvmStatus = NvMedia2DRegisterNvSciBufObj(m_up2DDevice.get(), m_srcBufObjs[packetIndex]);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DRegisterNvSciBufObj");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CStitchingConsumer::RegisterSignalSyncObj(NvSciSyncObj signalSyncObj)
{
    m_2DSignalSyncObj = signalSyncObj;
    auto nvmStatus = NvMedia2DRegisterNvSciSyncObj(m_up2DDevice.get(), NVMEDIA_EOFSYNCOBJ, signalSyncObj);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DRegisterNvSciSyncObj for EOF");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CStitchingConsumer::InsertPrefence(uint32_t packetIndex, NvSciSyncFence &prefence)
{
    auto nvmStatus = NvMedia2DInsertPreNvSciSyncFence(m_up2DDevice.get(), m_params, &prefence);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DInsertPreNvSciSyncFence");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CStitchingConsumer::RegisterWaiterSyncObj(NvSciSyncObj waiterSyncObj)
{
    auto nvmStatus = NvMedia2DRegisterNvSciSyncObj(m_up2DDevice.get(), NVMEDIA_PRESYNCOBJ, waiterSyncObj);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DRegisterNvSciSyncObj for PRE");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CStitchingConsumer::SetEofSyncObj(void)
{
    auto nvmStatus = NvMedia2DSetNvSciSyncObjforEOF(m_up2DDevice.get(), m_params, m_2DSignalSyncObj);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DSetNvSciSyncObjforEOF");

    return NVSIPL_STATUS_OK;
}

// Streaming functions
SIPLStatus CStitchingConsumer::ProcessPayload(uint32_t packetIndex, NvSciSyncFence *pPostfence)
{
    PLOG_DBG("Process payload (packetIndex = 0x%x).\n", packetIndex);

    NvMedia2DComposeResult composeResult;
    auto nvmStatus = NvMedia2DCompose(m_up2DDevice.get(), m_params, &composeResult);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DCompose");

    /* Get the end-of-frame fence for the compose operation */
    nvmStatus = NvMedia2DGetEOFNvSciSyncFence(m_up2DDevice.get(), &composeResult, pPostfence);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DGetEOFNvSciSyncFence");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CStitchingConsumer::OnProcessPayloadDone(uint32_t packetIndex)
{
    return NVSIPL_STATUS_OK;
}

SIPLStatus CStitchingConsumer::HandlePayload(void)
{
    NvSciStreamCookie cookie;

    /* Obtain packet with the new payload */
    auto sciErr = NvSciStreamConsumerPacketAcquire(m_handle, &cookie);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamConsumerPacketAcquire");
    PLOG_DBG("Acquired a packet (cookie = %u).\n", cookie);

    uint32_t packetId = 0U;
    auto status = GetIndexFromCookie(cookie, packetId);
    PCHK_STATUS_AND_RETURN(status, "PacketCookie2Id");

    ClientPacket *packet = GetPacketByCookie(cookie);
    PCHK_PTR_AND_RETURN(packet, "GetPacketByCookie");

    if (m_pProfiler != nullptr) {
        m_pProfiler->OnFrameAvailable();
    }

    m_frameNum++;

    CDisplayProducer::BufferInfo *bufInfo = m_spDisplayProd->GetBufferInfo(m_uSensorId);

    // If getting the null buffer info, it means there is no avaiable buffer.
    if (nullptr == bufInfo) {
        /* Release the packet back to the producer */
        sciErr = NvSciStreamConsumerPacketRelease(m_handle, packet->handle);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamConsumerPacketRelease");

        PLOG_WARN("CStitchingConsumer, no avaiable destination buffer!\n");
        return NVSIPL_STATUS_OK;
    }

    status = Set2DParameter(packetId, bufInfo->GetNvSciBufObj());
    PCHK_STATUS_AND_RETURN(status, "Set2DParameter");

    uint32_t elementId = 0U;

    /* If the received waiter obj if NULL,
    * the producer is done writing data into this element, skip waiting on pre-fence.
    * For consumer, there is only one waiter, using index 0 as default.
    */
    for (; elementId < m_elemsInfo.size(); ++elementId) {
        if (m_waiterSyncObjs[0U][elementId] != nullptr) {

            PLOG_DBG("Get prefence and insert it, waiter sync objects = %x\n", m_waiterSyncObjs[0U][elementId]);

            NvSciSyncFence prefence = NvSciSyncFenceInitializer;
            /* Query fences for this element from producer */
            sciErr = NvSciStreamBlockPacketFenceGet(m_handle, packet->handle, 0U, elementId, &prefence);
            PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockPacketFenceGet");

            status = InsertPrefence(packetId, prefence);
            NvSciSyncFenceClear(&prefence);
            PCHK_STATUS_AND_RETURN(status, ": InsertPrefence");

            break;
        }
    }

    status = SetEofSyncObj();
    PCHK_STATUS_AND_RETURN(status, "SetEofSyncObj");

    // If the destination buffer is not ready, then wait for the dst buffer available.
    // Do not clear the shared prefence of display buffer here!
    // Clear this prefence by display producer.
    if (m_cpuWaitContext != nullptr) {
        const NvSciSyncFence &prefence = bufInfo->GetPreFence();
        auto sciErr = NvSciSyncFenceWait(&prefence, m_cpuWaitContext, FENCE_FRAME_TIMEOUT_US);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncFenceWait");
    }

    NvSciSyncFence postfence = NvSciSyncFenceInitializer;
    status = ProcessPayload(packetId, &postfence);
    PCHK_STATUS_AND_RETURN(status, "ProcessPayload");

    status = m_spDisplayProd->Submit(m_uSensorId, &postfence);
    PCHK_STATUS_AND_RETURN(status, "Submit");

    sciErr = NvSciStreamBlockPacketFenceSet(m_handle, packet->handle, elementId, &postfence);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockPacketFenceSet");
    NvSciSyncFenceClear(&postfence);

    /* Release the packet back to the producer */
    sciErr = NvSciStreamConsumerPacketRelease(m_handle, packet->handle);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamConsumerPacketRelease");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CStitchingConsumer::Set2DParameter(uint32_t packetId, NvSciBufObj bufObj)
{
    auto nvmStatus = NvMedia2DGetComposeParameters(m_up2DDevice.get(), &m_params);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DGetComposeParameters");

    /* Set the source layer parameters for layer zero */
    nvmStatus = NvMedia2DSetSrcNvSciBufObj(m_up2DDevice.get(), m_params, 0U, m_srcBufObjs[packetId]);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DSetSrcNvSciBufObj");

    nvmStatus = NvMedia2DSetSrcGeometry(m_up2DDevice.get(), m_params, 0U, NULL, &m_spDisplayProd->GetRect(m_uSensorId),
                                        NVMEDIA_2D_TRANSFORM_NONE);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DSetSrcGeometry");

    /* Set the destination surface */
    nvmStatus = NvMedia2DSetDstNvSciBufObj(m_up2DDevice.get(), m_params, bufObj);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DSetDstNvSciBufObj");

    return NVSIPL_STATUS_OK;
}
