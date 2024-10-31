// Copyright (c) 2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include <algorithm>

#include "CStitchingConsumer.hpp"
#include "CDisplayProducer.hpp"
#include "nvscibuf.h"

constexpr static uint32_t PLANE_COUNT = 1U;
constexpr static uint32_t PLANE_BASEADDR_ALIGN = 256U;

CDisplayProducer::CDisplayProducer(NvSciStreamBlock handle)
    : CProducer("CDisplayProducer", handle, 0U)
{
}

void CDisplayProducer::PreInit(uint32_t numPackets, uint32_t width, uint32_t height)
{
    m_vBufInfo.resize(numPackets);
    m_uWidth = width;
    m_uHeight = height;
}

CDisplayProducer::~CDisplayProducer()
{
    if (m_displayCPUWaitCtx != nullptr) {
        NvSciSyncCpuWaitContextFree(m_displayCPUWaitCtx);
        m_displayCPUWaitCtx = nullptr;
    }

    LOG_DBG("CDisplayProducer released.\n");
}

SIPLStatus CDisplayProducer::Deinit()
{
    LOG_DBG("CDisplayProducer Deinit start.\n");

    {
        std::unique_lock<std::mutex> lk(m_mutex);

        m_bIsRunning = false;
    }

    m_conditionVar.notify_one();

    if (m_upthread->joinable()) {
        m_upthread->join();
    }

    LOG_DBG("CDisplayProducer Deinit end.\n");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CDisplayProducer::HandleClientInit()
{
    m_bIsRunning = true;
    m_upthread.reset(new std::thread(&CDisplayProducer::DoWork, this));
    if (m_upthread == nullptr) {
        m_bIsRunning = false;
        LOG_ERR("Failed to create compositor thread\n");
        return NVSIPL_STATUS_OUT_OF_MEMORY;
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CDisplayProducer::SetDataBufAttrList(NvSciBufAttrList &bufAttrList)
{
    if (!m_vCompositorIds.empty()) {
        // The compositors should have the same attribute list, and we get the first one from them.
        NvSciError sciErr = NvSciBufAttrListClone(
            m_compositorInfo[m_vCompositorIds[0U]].compositor->GetBufferAttributList(), &bufAttrList);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListClone");
    }

    NvSciBufType bufType = NvSciBufType_Image;
    bool imgCpuAccess = false;
    uint32_t planeCount = PLANE_COUNT;
    NvSciBufAttrValColorFmt planeColorFmt = NvSciColor_A8B8G8R8;
    NvSciBufAttrValColorStd planeColorStd = NvSciColorStd_SRGB;
    NvSciBufAttrValImageLayoutType imgLayout = NvSciBufImage_PitchLinearType;
    uint64_t topPadding = 0U;
    uint64_t bottomPadding = 0U;
    uint64_t leftPadding = 0U;
    uint64_t rightPadding = 0U;
    uint32_t planeWidth = m_uWidth;
    uint32_t planeHeight = m_uHeight;
    uint32_t planeBaseAddrAlign = PLANE_BASEADDR_ALIGN;
    NvSciBufAttrKeyValuePair keyVals[] = {
        { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(NvSciBufType) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &imgCpuAccess, sizeof(bool) },
        { NvSciBufImageAttrKey_PlaneCount, &planeCount, sizeof(uint32_t) },
        { NvSciBufImageAttrKey_PlaneColorFormat, &planeColorFmt, sizeof(NvSciBufAttrValColorFmt) },
        { NvSciBufImageAttrKey_PlaneColorStd, &planeColorStd, sizeof(NvSciBufAttrValColorStd) },
        { NvSciBufImageAttrKey_Layout, &imgLayout, sizeof(NvSciBufAttrValImageLayoutType) },
        { NvSciBufImageAttrKey_TopPadding, &topPadding, sizeof(uint64_t) },
        { NvSciBufImageAttrKey_BottomPadding, &bottomPadding, sizeof(uint64_t) },
        { NvSciBufImageAttrKey_LeftPadding, &leftPadding, sizeof(uint64_t) },
        { NvSciBufImageAttrKey_RightPadding, &rightPadding, sizeof(uint64_t) },
        { NvSciBufImageAttrKey_PlaneWidth, &planeWidth, sizeof(uint32_t) },
        { NvSciBufImageAttrKey_PlaneHeight, &planeHeight, sizeof(uint32_t) },
        { NvSciBufImageAttrKey_PlaneBaseAddrAlign, &planeBaseAddrAlign, sizeof(uint32_t) }
    };

    size_t length = sizeof(keyVals) / sizeof(NvSciBufAttrKeyValuePair);
    auto sciErr = NvSciBufAttrListSetAttrs(bufAttrList, keyVals, length);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CDisplayProducer::SetSyncAttrList(NvSciSyncAttrList &signalerAttrList, NvSciSyncAttrList &waiterAttrList)
{
    auto status = SetCpuSyncAttrList(waiterAttrList, NvSciSyncAccessPerm_WaitOnly, true);
    PCHK_STATUS_AND_RETURN(status, "SetCpuSyncAttrList");

    /* Create a context for CPU waiting */
    auto sciErr = NvSciSyncCpuWaitContextAlloc(m_sciSyncModule, &m_displayCPUWaitCtx);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncCpuWaitContextAlloc");

    status = SetCpuSyncAttrList(signalerAttrList, NvSciSyncAccessPerm_SignalOnly, true);
    PCHK_STATUS_AND_RETURN(status, "SetCpuSyncAttrList");

    return NVSIPL_STATUS_OK;
}

void CDisplayProducer::OnPacketGotten(uint32_t packetIndex)
{
    {
        std::unique_lock<std::mutex> lck(m_mutex);
        m_vBufInfo[packetIndex].m_bIsAvaiable = true;
    }

    LOG_DBG("CDisplayProducer::OnPacketGotten, packet id = %u \n", packetIndex);
}

SIPLStatus CDisplayProducer::RegisterSignalSyncObj(NvSciSyncObj signalSyncObj)
{
    m_cpuSignalSyncObj = signalSyncObj;
    return NVSIPL_STATUS_OK;
}

SIPLStatus CDisplayProducer::RegisterWaiterSyncObj(NvSciSyncObj waiterSyncObj)
{
    return NVSIPL_STATUS_OK;
}

SIPLStatus CDisplayProducer::MapDataBuffer(uint32_t packetIndex, NvSciBufObj bufObj)
{
    LOG_DBG("CDisplayProducer::MapDataBuffer, packet id = %u \n", packetIndex);
    CDisplayProducer::BufferInfo &bufInfo = m_vBufInfo[packetIndex];

    auto sciErr = NvSciBufObjDup(bufObj, &bufInfo.m_bufObj);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjDup");

    for (auto id : m_vCompositorIds) {
        auto status = m_compositorInfo[id].compositor->RegisterDstNvSciObjBuf(bufInfo.m_bufObj);
        PCHK_STATUS_AND_RETURN(status, "RegisterDstNvSciObjBuf");
    }

    bufInfo.m_bIsAvaiable = true;

    return NVSIPL_STATUS_OK;
}

SIPLStatus CDisplayProducer::MapMetaBuffer(uint32_t packetIndex, NvSciBufObj bufObj)
{
    return NVSIPL_STATUS_OK;
}

SIPLStatus CDisplayProducer::InsertPrefence(uint32_t packetIndex, NvSciSyncFence &prefence)
{
    LOG_DBG("CDisplayProducer::InsertPrefence, packet id = %u \n", packetIndex);

    // Need to free previous resources for this fence and then duplicate the prefence.
    NvSciSyncFenceClear(&m_vBufInfo[packetIndex].m_prefence);
    auto sciErr = NvSciSyncFenceDup(&prefence, &m_vBufInfo[packetIndex].m_prefence);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncFenceDup");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CDisplayProducer::GetPostfence(uint32_t packetIndex, NvSciSyncFence *pPostfence)
{
    auto sciErr = NvSciSyncObjGenerateFence(m_cpuSignalSyncObj, pPostfence);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncObjGenerateFence prefence");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CDisplayProducer::MapPayload(void *pBuffer, uint32_t &packetIndex)
{
    NvSciBufObj dstObj = *(static_cast<NvSciBufObj *>(pBuffer));
    std::vector<BufferInfo>::iterator it =
        std::find_if(m_vBufInfo.begin(), m_vBufInfo.end(),
                     [dstObj](const BufferInfo &bufInfo) { return (dstObj == bufInfo.m_bufObj); });

    if (m_vBufInfo.end() == it) {
        // Didn't find matching buffer
        PLOG_ERR("MapPayload, failed to get packet index for buffer\n");
        return NVSIPL_STATUS_ERROR;
    }

    packetIndex = std::distance(m_vBufInfo.begin(), it);

    return NVSIPL_STATUS_OK;
}

void CDisplayProducer::RegisterCompositor(uint32_t uCompositorId, CStitchingConsumer *compositor)
{
    m_vCompositorIds.push_back(uCompositorId);
    m_compositorInfo[uCompositorId].compositor = compositor;
}

CDisplayProducer::BufferInfo *CDisplayProducer::GetBufferInfo(uint32_t uCompositorId)
{
    uint32_t index = 0U;
    {
        std::unique_lock<std::mutex> lck(m_mutex);
        CDisplayProducer::BufferInfo &bufInfo = m_vBufInfo[m_uCurBufInfoIndex];
        if (!bufInfo.m_bIsAvaiable) {
            LOG_WARN("CDisplayProducer::GetBufferInfo, no avaiable buffer!");
            return nullptr;
        }

        // Ensure that the latest operation is submitted for this display buffer.
        // If this compositor has submitted operation,
        // we need to reset the submit status and clear postfence.
        if (m_compositorInfo[uCompositorId].hasSubmitted) {
            m_compositorInfo[uCompositorId].hasSubmitted = false;
            NvSciSyncFenceClear(&bufInfo.m_postfences[uCompositorId]);
        }

        index = m_uCurBufInfoIndex;
    }

    return &m_vBufInfo[index];
}

SIPLStatus CDisplayProducer::ComputeInputRects(void)
{
    // Set up the destination rectangles
    uint16_t xStep = m_uWidth;
    uint16_t yStep = m_uHeight;
    uint16_t countPerLine = 1U;
    uint32_t cameraCount = m_vCompositorIds.size();
    if (cameraCount > 1U) {
        if (cameraCount <= 4U) {
            countPerLine = 2U;
        } else {
            countPerLine = 4U;
        }
    }
    xStep = m_uWidth / countPerLine;
    yStep = m_uHeight / countPerLine;
    for (auto i = 0U; i < cameraCount; ++i) {
        auto rowIndex = i / countPerLine;
        auto colIndex = i % countPerLine;
        uint16_t startx = colIndex * xStep;
        uint16_t starty = rowIndex * yStep;
        uint16_t endx = startx + xStep;
        uint16_t endy = starty + yStep;
        m_oInputRects[m_vCompositorIds[i]] = { startx, starty, endx, endy };
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CDisplayProducer::Submit(uint32_t uCompositorId, const NvSciSyncFence *fence)
{
    LOG_DBG("CDisplayProducer::Submit \n");

    {
        std::unique_lock<std::mutex> lck(m_mutex);
        CDisplayProducer::BufferInfo &bufInfo = m_vBufInfo[m_uCurBufInfoIndex];

        auto sciErr = NvSciSyncFenceDup(fence, &bufInfo.m_postfences[uCompositorId]);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncFenceDup");

        // Check the compositors the status of submit,
        // and if all compositors has submitted operation,
        // we notify work thread to post display buffer to consumer.
        m_compositorInfo[uCompositorId].hasSubmitted = true;
        for (auto id : m_vCompositorIds) {
            if (!m_compositorInfo[id].hasSubmitted) {
                LOG_DBG("CDisplayProducer::Submit, no submit for compositor id = %u\n", uCompositorId);
                return NVSIPL_STATUS_OK;
            }
        }

        // To clear submit status and post the buffer to display.
        for (auto id : m_vCompositorIds) {
            m_compositorInfo[id].hasSubmitted = false;
        }

        bufInfo.m_bIsAvaiable = false;
        m_postBufInfoIds.push_back(m_uCurBufInfoIndex++);
        m_uCurBufInfoIndex %= m_numPacket;
    }

    m_conditionVar.notify_one();

    LOG_DBG("CDisplayProducer::notify to display \n");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CDisplayProducer::HandleSetupComplete(void)
{
    auto status = ComputeInputRects();
    PCHK_STATUS_AND_RETURN(status, "ComputeInputRects");

    status = CProducer::HandleSetupComplete();
    PCHK_STATUS_AND_RETURN(status, "HandleSetupComplete");

    return NVSIPL_STATUS_OK;
}

void CDisplayProducer::DoWork()
{
    pthread_setname_np(pthread_self(), "CDisplayProducer");
    LOG_DBG("CDisplayProducer::DoWork, start! \n");

    while (1) {
        uint32_t bufId = 0U;
        {
            std::unique_lock<std::mutex> lk(m_mutex);
            m_conditionVar.wait(lk, [this] { return !(m_postBufInfoIds.empty() && m_bIsRunning); });

            if (!m_bIsRunning) {
                LOG_DBG("CDisplayProducer::DoWork, exit! \n");
                return;
            }

            if (m_postBufInfoIds.empty()) {
                LOG_WARN("CDisplayProducer::DoWork, empty m_postBufInfoIds! \n");
                continue;
            }

            bufId = m_postBufInfoIds.front();
            m_postBufInfoIds.erase(m_postBufInfoIds.begin());
        }

        LOG_DBG("CDisplayProducer::DoWork, post packet to display!, packet id = %d \n", bufId);

        CDisplayProducer::BufferInfo &bufInfo = m_vBufInfo[bufId];

        // post packet to consumer.
        auto status = Post(&bufInfo.m_bufObj);
        if (status != NVSIPL_STATUS_OK) {
            LOG_DBG("CDisplayProducer::DoWork, post packet to display. Error! packet id = %d \n", bufId);
            break;
        }

        // Wait for all compositors to compelete the operation.
        for (auto compId : m_vCompositorIds) {
            auto sciErr =
                NvSciSyncFenceWait(&bufInfo.m_postfences[compId], m_displayCPUWaitCtx, FENCE_FRAME_TIMEOUT_US);
            NvSciSyncFenceClear(&bufInfo.m_postfences[compId]);
            if (sciErr != NvSciError_Success) {
                LOG_ERR("CDisplayProducer::DoWork, NvSciSyncFenceWait error! \n");
                break;
            }
        }

        // All compositors has compeleted the operation and then signal the sync object.
        auto sciErr = NvSciSyncObjSignal(m_cpuSignalSyncObj);
        if (sciErr != NvSciError_Success) {
            LOG_ERR("CDisplayProducer::DoWork, signal fence error! \n");
            break;
        }
    } // while ()

    LOG_DBG("CDisplayProducer::DoWork, end! \n");
}
