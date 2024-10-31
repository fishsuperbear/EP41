// Copyright (c) 2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include "CDisplayConsumer.hpp"
#include "nvscibuf.h"

namespace hozon {
namespace netaos {
namespace desay {

constexpr static uint32_t DISPLAY_WIDTH = 1920U;
constexpr static uint32_t DISPLAY_HEIGHT = 1536U;

CDisplayConsumer::CDisplayConsumer(NvSciStreamBlock handle, uint32_t uSensorId, NvSciStreamBlock queueHandle)
    : CConsumer("CDisplayConsumer", handle, uSensorId, queueHandle)
{
    m_inited = false;
}

void CDisplayConsumer::PreInit(const std::shared_ptr<COpenWFDController> &wfdController, uint32_t wfdPipelineId)
{
    m_spWFDController = wfdController;
    m_wfdPipelineId = wfdPipelineId;
}

SIPLStatus CDisplayConsumer::HandleClientInit()
{
    return NVSIPL_STATUS_OK;
}

CDisplayConsumer::~CDisplayConsumer(void)
{
    m_inited = false;

    LOG_DBG("CDisplayConsumer released.\n");
}

// Buffer setup functions
SIPLStatus CDisplayConsumer::SetDataBufAttrList(NvSciBufAttrList &bufAttrList)
{
    auto status = m_spWFDController->SetDisplayNvSciBufAttributesNVX(bufAttrList);
    PCHK_STATUS_AND_RETURN(status, "SetDataBufAttrList");

    return NVSIPL_STATUS_OK;
}

// Sync object setup functions
SIPLStatus CDisplayConsumer::SetSyncAttrList(NvSciSyncAttrList &signalerAttrList, NvSciSyncAttrList &waiterAttrList)
{
    auto status = m_spWFDController->SetDisplayNvSciSyncAttributesNVX(signalerAttrList, waiterAttrList);
    PCHK_STATUS_AND_RETURN(status, "SetSyncAttrList");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CDisplayConsumer::MapDataBuffer(uint32_t packetIndex, NvSciBufObj bufObj)
{
    auto status = PopulateBufAttr(bufObj, m_bufAttrs[packetIndex]);
    PCHK_STATUS_AND_RETURN(status, "PopulateBufAttr");

    LOG_INFO("CDisplayConsumer::MapDataBuffer, display source width = %u, height = %u \n", m_bufAttrs[0].planeWidths[0],
             m_bufAttrs[0].planeHeights[0]);

    status = m_spWFDController->CreateWFDSource(bufObj, m_wfdPipelineId, packetIndex);
    PCHK_STATUS_AND_RETURN(status, "CreateWFDSource");

    LOG_DBG("%s: Create wfd source success\n", __func__);

    return NVSIPL_STATUS_OK;
}

SIPLStatus CDisplayConsumer::RegisterSignalSyncObj(NvSciSyncObj signalSyncObj)
{
    LOG_DBG("CDisplayConsumer::RegisterSignalSyncObj \n");

    auto status = m_spWFDController->RegisterSignalSyncObj(signalSyncObj, m_wfdPipelineId);
    PCHK_STATUS_AND_RETURN(status, "RegisterSignalSyncObj");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CDisplayConsumer::InsertPrefence(uint32_t packetIndex, NvSciSyncFence &prefence)
{
    auto status = m_spWFDController->InsertPrefence(m_wfdPipelineId, packetIndex, prefence);
    PCHK_STATUS_AND_RETURN(status, "RegisterSignalSyncObj");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CDisplayConsumer::RegisterWaiterSyncObj(NvSciSyncObj waiterSyncObj)
{
    return NVSIPL_STATUS_OK;
}

SIPLStatus CDisplayConsumer::HandleSetupComplete(void)
{
    auto status = CConsumer::HandleSetupComplete();
    PCHK_STATUS_AND_RETURN(status, "CConsumer::HandleSetupComplete");
    // LOG_MSG("source width = %d, source height = %d \n", m_bufAttrs[0].planeWidths[0], m_bufAttrs[0].planeHeights[0]);
    uint32_t sourceX = 0, sourceY = 0;
    uint32_t sourceWidth = m_bufAttrs[0].planeWidths[0];
    uint32_t sourceHeight = m_bufAttrs[0].planeHeights[0];
    uint32_t dstWidth = DISPLAY_WIDTH;
    uint32_t dstHeight = DISPLAY_HEIGHT;

    LOG_INFO("display config bcropFlag:%d bzoomFlag:%d\n", m_dispConfig.bcropFlag, m_dispConfig.bzoomFlag);
    if (m_dispConfig.bcropFlag) {
        sourceX = m_dispConfig.crop_x;
        sourceY = m_dispConfig.crop_y;
        sourceWidth = m_dispConfig.crop_width;
        sourceHeight = m_dispConfig.crop_height;
    }

    if (m_dispConfig.bzoomFlag) {
        dstWidth = m_dispConfig.zoom_width;
        dstHeight = m_dispConfig.zoom_height;
    }

    LOG_INFO("source x:%d y:%d width:%d height:%d\n", sourceX, sourceY, sourceWidth, sourceHeight);
    LOG_INFO("destination width:%d height:%d\n", dstWidth, dstHeight);

    status = m_spWFDController->SetRect(sourceX, sourceY, sourceWidth, sourceHeight, dstWidth, dstHeight, m_wfdPipelineId);
    PCHK_STATUS_AND_RETURN(status, "SetRect");

    m_inited = true;

    return NVSIPL_STATUS_OK;
}

SIPLStatus CDisplayConsumer::SetEofSyncObj(void)
{
    return NVSIPL_STATUS_OK;
}

// Streaming functions
SIPLStatus CDisplayConsumer::ProcessPayload(uint32_t packetIndex, NvSciSyncFence *pPostfence)
{
    if (!m_inited) {
        LOG_ERR("Post before init completed.\n");
        return NVSIPL_STATUS_INVALID_STATE;
    }

    LOG_DBG("CDisplayConsumer::ProcessPayload, packet id = %u\n", packetIndex);

    auto status = m_spWFDController->Flip(m_wfdPipelineId, packetIndex, pPostfence);
    PCHK_STATUS_AND_RETURN(status, "Flip");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CDisplayConsumer::OnProcessPayloadDone(uint32_t packetIndex)
{
    return NVSIPL_STATUS_OK;
}

}
}
}