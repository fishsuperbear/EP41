// Copyright (c) 2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include "COpenWFDController.hpp"
#include "nvscibuf.h"

constexpr static uint32_t WFD_PORT_MODES_COUNT = 1U;
constexpr static uint32_t WFD_PIPELINE_MAIN_LAYER_IDX = 0U;

COpenWFDController::COpenWFDController()
{
    m_name = "COpenWFDController";

    for (uint32_t i = 0U; i < MAX_NUM_WFD_PIPELINES; ++i) {
        for (uint32_t j = 0U; j < MAX_NUM_PACKETS; ++j) {
            m_wfdSources[i][j] = WFD_INVALID_HANDLE;
        }
    }
}

// In this sample, we just cover two use cases.
// For use case of DM-MST display, there are two ports and two pipelines being used.
// For use case of stitching cameras display, there is only one port and one pipeline being used.
const SIPLStatus COpenWFDController::InitResource(uint32_t numPorts)
{
    if (numPorts > MAX_NUM_WFD_PORTS) {
        LOG_ERR("COpenWFDController, the maxinum number of ports is %d \n", MAX_NUM_WFD_PORTS);
        return NVSIPL_STATUS_RESOURCE_ERROR;
    }

    m_wfdDevice = wfdCreateDevice(WFD_DEFAULT_DEVICE_ID, NULL);
    if (!m_wfdDevice) {
        LOG_ERR("wfdCreateDevice failed\n");
        return NVSIPL_STATUS_RESOURCE_ERROR;
    }

    m_wfdNumPorts = wfdEnumeratePorts(m_wfdDevice, m_wfdPortIds, numPorts, NULL);
    if (m_wfdNumPorts < 1 || static_cast<uint32_t>(m_wfdNumPorts) != numPorts) {
        PGET_WFDERROR_AND_RETURN(m_wfdDevice);
    }

    LOG_MSG("COpenWFDController, wfd number of ports = %d \n", m_wfdNumPorts);

    for (uint32_t i = 0U; i < static_cast<uint32_t>(m_wfdNumPorts); ++i) {

        LOG_MSG("COpenWFDController, port id = %d \n", m_wfdPortIds[i]);

        m_wfdPorts[i] = wfdCreatePort(m_wfdDevice, m_wfdPortIds[i], NULL);
        if (!m_wfdPorts[i]) {
            PGET_WFDERROR_AND_RETURN(m_wfdDevice);
        }

        WFDPortMode wfdPortMode = WFD_INVALID_HANDLE;
        auto wfdNumModes = wfdGetPortModes(m_wfdDevice, m_wfdPorts[i], &wfdPortMode, WFD_PORT_MODES_COUNT);
        if (!wfdNumModes) {
            PGET_WFDERROR_AND_RETURN(m_wfdDevice);
        }

        m_windowWidths[i] = wfdGetPortModeAttribi(m_wfdDevice, m_wfdPorts[i], wfdPortMode, WFD_PORT_MODE_WIDTH);
        m_windowHeights[i] = wfdGetPortModeAttribi(m_wfdDevice, m_wfdPorts[i], wfdPortMode, WFD_PORT_MODE_HEIGHT);

        LOG_MSG("window width = %d, window height = %d \n", m_windowWidths[i], m_windowHeights[i]);

        wfdSetPortMode(m_wfdDevice, m_wfdPorts[i], wfdPortMode);
        PGET_WFDERROR_AND_RETURN(m_wfdDevice);

        wfdDeviceCommit(m_wfdDevice, WFD_COMMIT_ENTIRE_PORT, m_wfdPorts[i]);
        PGET_WFDERROR_AND_RETURN(m_wfdDevice);

        auto status = CreatePipeline(m_wfdPorts[i], i);
        PCHK_STATUS_AND_RETURN(status, "CreatePipeline");
    }

    return NVSIPL_STATUS_OK;
}

const SIPLStatus COpenWFDController::CreatePipeline(WFDPort port, uint32_t wfdPipelineIdx)
{
    if (m_wfdNumPipelines >= MAX_NUM_WFD_PIPELINES) {
        LOG_ERR("COpenWFDController, can not create more pipelines, maybe you can adjust the max number of pipelines "
                "if support \n",
                MAX_NUM_WFD_PIPELINES);
        return NVSIPL_STATUS_ERROR;
    }

    //Get the number of bindable pipeline IDs for a port
    auto wfdNumPipelines = wfdGetPortAttribi(m_wfdDevice, port, WFD_PORT_PIPELINE_ID_COUNT);
    PGET_WFDERROR_AND_RETURN(m_wfdDevice);

    //Populate pipeline IDs into m_wfdPipelines
    std::vector<WFDint> wfdBindablePipeIds(wfdNumPipelines);
    wfdGetPortAttribiv(m_wfdDevice, port, WFD_PORT_BINDABLE_PIPELINE_IDS, wfdNumPipelines, wfdBindablePipeIds.data());
    PGET_WFDERROR_AND_RETURN(m_wfdDevice);
    if (wfdNumPipelines <= 0) {
        LOG_ERR("InitWFD, no pipeline is found.");
        return NVSIPL_STATUS_ERROR;
    }

    LOG_MSG("COpenWFDController, wfd number of pipelines = %d, wfdBindablePipeIds = %d\n", wfdNumPipelines,
            wfdBindablePipeIds[1]);

    m_wfdPipelines[wfdPipelineIdx] =
        wfdCreatePipeline(m_wfdDevice, wfdBindablePipeIds[WFD_PIPELINE_MAIN_LAYER_IDX], NULL);
    if (!m_wfdPipelines) {
        PGET_WFDERROR_AND_RETURN(m_wfdDevice);
    }

    wfdBindPipelineToPort(m_wfdDevice, port, m_wfdPipelines[wfdPipelineIdx]);
    PGET_WFDERROR_AND_RETURN(m_wfdDevice);

    LOG_DBG("%s: pipeline is created and bound successfully\n", __func__);

    wfdDeviceCommit(m_wfdDevice, WFD_COMMIT_ENTIRE_PORT, port);
    PGET_WFDERROR_AND_RETURN(m_wfdDevice);

    m_wfdNumPipelines++;

    LOG_DBG("%s: wfdBindPipelineToPort success\n", __func__);

    return NVSIPL_STATUS_OK;
}

COpenWFDController::~COpenWFDController(void)
{
    LOG_DBG("COpenWFDController: Release.");
}

const SIPLStatus COpenWFDController::DeInit(void)
{
    LOG_DBG("COpenWFDController: DeInit. \n");

    for (uint32_t i = 0U; i < m_wfdNumPipelines; ++i) {
        if (m_wfdDevice && m_wfdPipelines[i]) {
            // Perform a null flip
            wfdBindSourceToPipeline(m_wfdDevice, m_wfdPipelines[i], (WFDSource)0, WFD_TRANSITION_AT_VSYNC, NULL);
            CHECK_WFD_ERROR(m_wfdDevice);

            wfdSetPipelineAttribi(m_wfdDevice, m_wfdPipelines[i],
                                  static_cast<WFDPipelineConfigAttrib>(WFD_PIPELINE_COMMIT_NON_BLOCKING_NVX),
                                  WFD_FALSE);
            CHECK_WFD_ERROR(m_wfdDevice);

            wfdDeviceCommit(m_wfdDevice, WFD_COMMIT_PIPELINE, m_wfdPipelines[i]);
            CHECK_WFD_ERROR(m_wfdDevice);

            wfdDeviceCommit(m_wfdDevice, WFD_COMMIT_PIPELINE, m_wfdPipelines[i]);
            CHECK_WFD_ERROR(m_wfdDevice);
        }
    }

    for (uint32_t i = 0U; i < m_wfdNumPipelines; ++i) {

        LOG_INFO("COpenWFDController, wfdDestroySource \n");

        for (uint32_t j = 0U; j < MAX_NUM_PACKETS; ++j) {
            if (m_wfdSources[i][j] != WFD_INVALID_HANDLE) {
                wfdDestroySource(m_wfdDevice, m_wfdSources[i][j]);
                CHECK_WFD_ERROR(m_wfdDevice);
            }
        }

        LOG_INFO("COpenWFDController, wfdDestroyPipeline \n");
        if (m_wfdPipelines[i]) {
            wfdDestroyPipeline(m_wfdDevice, m_wfdPipelines[i]);
            CHECK_WFD_ERROR(m_wfdDevice);
        }
    }

    for (uint32_t i = 0U; i < static_cast<uint32_t>(m_wfdNumPorts); ++i) {
        LOG_INFO("COpenWFDController, wfdDestroyPort \n");
        if (m_wfdPorts[i]) {
            wfdDestroyPort(m_wfdDevice, m_wfdPorts[i]);
            CHECK_WFD_ERROR(m_wfdDevice);
        }
    }

    LOG_INFO("COpenWFDController, wfdDestroyDevice \n");

    if (m_wfdDevice) {
        wfdDestroyDevice(m_wfdDevice);
    }

    LOG_DBG("COpenWFDController DeInit end.\n");

    return NVSIPL_STATUS_OK;
}

// Buffer setup functions
const SIPLStatus COpenWFDController::SetDisplayNvSciBufAttributesNVX(NvSciBufAttrList &attrList) const
{
    // Default buffer attributes
    NvSciBufType bufType = NvSciBufType_Image;
    NvSciBufAttrValImageScanType bufScanType = NvSciBufScan_ProgressiveType;
    NvSciBufAttrValAccessPerm bufPerm = NvSciBufAccessPerm_Readonly;

    bool needCpuAccessFlag = false;
    NvSciBufAttrKeyValuePair bufAttrs[] = {
        { NvSciBufGeneralAttrKey_RequiredPerm, &bufPerm, sizeof(bufPerm) },
        { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &needCpuAccessFlag, sizeof(needCpuAccessFlag) },
        { NvSciBufImageAttrKey_ScanType, &bufScanType, sizeof(bufScanType) },
    };

    WFDErrorCode wfdErr = wfdNvSciBufSetDisplayAttributesNVX(&attrList);
    PCHK_WFDSTATUS_AND_RETURN(wfdErr, "wfdNvSciBufSetDisplayAttributesNVX");

    auto sciErr = NvSciBufAttrListSetAttrs(attrList, bufAttrs, sizeof(bufAttrs) / sizeof(NvSciBufAttrKeyValuePair));
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");

    return NVSIPL_STATUS_OK;
}

// Sync object setup functions
const SIPLStatus COpenWFDController::SetDisplayNvSciSyncAttributesNVX(NvSciSyncAttrList &signalerAttrList,
                                                                      NvSciSyncAttrList &waiterAttrList) const
{
    WFDErrorCode wfdErr = wfdNvSciSyncSetWaiterAttributesNVX(&waiterAttrList);
    PCHK_WFDSTATUS_AND_RETURN(wfdErr, "wfdNvSciSyncSetWaiterAttributesNVX");

    NvSciSyncAccessPerm accessPerm = NvSciSyncAccessPerm_WaitOnly;
    NvSciSyncAttrKeyValuePair syncWaiterAttrs[] = {
        { NvSciSyncAttrKey_RequiredPerm, (void *)&accessPerm, sizeof(accessPerm) },
    };

    auto sciErr = NvSciSyncAttrListSetAttrs(waiterAttrList, syncWaiterAttrs,
                                            sizeof(syncWaiterAttrs) / sizeof(NvSciSyncAttrKeyValuePair));
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListSetAttrs");

    wfdErr = wfdNvSciSyncSetSignalerAttributesNVX(&signalerAttrList);
    PCHK_WFDSTATUS_AND_RETURN(wfdErr, "wfdNvSciSyncSetSignalerAttributesNVX");

    bool cpuSync = true;
    accessPerm = NvSciSyncAccessPerm_SignalOnly;
    NvSciSyncAttrKeyValuePair syncSignalAttrs[] = {
        { NvSciSyncAttrKey_NeedCpuAccess, &cpuSync, sizeof(cpuSync) },
        { NvSciSyncAttrKey_RequiredPerm, (void *)&accessPerm, sizeof(accessPerm) },
    };

    sciErr = NvSciSyncAttrListSetAttrs(signalerAttrList, syncSignalAttrs,
                                       sizeof(syncSignalAttrs) / sizeof(NvSciSyncAttrKeyValuePair));
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListSetAttrs");

    return NVSIPL_STATUS_OK;
}

const SIPLStatus COpenWFDController::CreateWFDSource(NvSciBufObj &obj, uint32_t wfdPipelineIdx, uint32_t packetId)
{
    m_wfdSources[wfdPipelineIdx][packetId] =
        wfdCreateSourceFromNvSciBufNVX(m_wfdDevice, m_wfdPipelines[wfdPipelineIdx], &obj);
    if (!m_wfdSources[wfdPipelineIdx][packetId]) {
        PGET_WFDERROR_AND_RETURN(m_wfdDevice);
    }

    return NVSIPL_STATUS_OK;
}

const SIPLStatus COpenWFDController::RegisterSignalSyncObj(NvSciSyncObj syncObj, uint32_t wfdPipelineIdx)
{
    LOG_DBG("COpenWFDController::RegisterSignalSyncObj \n");

    WFDint ret = wfdGetPipelineAttribi(m_wfdDevice, m_wfdPipelines[wfdPipelineIdx],
                                       static_cast<WFDPipelineConfigAttrib>(WFD_PIPELINE_COMMIT_NON_BLOCKING_NVX));
    PGET_WFDERROR_AND_RETURN(m_wfdDevice);

    if (WFD_FALSE == ret) {
        LOG_INFO("CDisplayConsumer, openwfd non-blocking commit disabled, to enable it\n");
        wfdSetPipelineAttribi(m_wfdDevice, m_wfdPipelines[wfdPipelineIdx],
                              static_cast<WFDPipelineConfigAttrib>(WFD_PIPELINE_COMMIT_NON_BLOCKING_NVX), WFD_TRUE);
        PGET_WFDERROR_AND_RETURN(m_wfdDevice);
    }

    wfdSetPipelineAttribi(m_wfdDevice, m_wfdPipelines[wfdPipelineIdx],
                          static_cast<WFDPipelineConfigAttrib>(WFD_PIPELINE_POSTFENCE_SCANOUT_BEGIN_NVX), WFD_FALSE);
    PGET_WFDERROR_AND_RETURN(m_wfdDevice);

    // only one signal object can be registered.
    if (wfdPipelineIdx < 1) {
        WFDErrorCode wfdErr = wfdRegisterPostFlipNvSciSyncObjNVX(m_wfdDevice, &syncObj);
        PCHK_WFDSTATUS_AND_RETURN(wfdErr, "wfdRegisterPostFlipNvSciSyncObjNVX");
    }

    return NVSIPL_STATUS_OK;
}

const SIPLStatus
COpenWFDController::InsertPrefence(uint32_t wfdPipelineIdx, uint32_t packetIndex, NvSciSyncFence &prefence) const
{
    WFDErrorCode wfdErr =
        wfdBindNvSciSyncFenceToSourceNVX(m_wfdDevice, m_wfdSources[wfdPipelineIdx][packetIndex], &prefence);
    PCHK_WFDSTATUS_AND_RETURN(wfdErr, "wfdBindNvSciSyncFenceToSourceNVX");

    return NVSIPL_STATUS_OK;
}

const SIPLStatus COpenWFDController::RegisterWaiterSyncObj(NvSciSyncObj waiterSyncObj) const
{
    return NVSIPL_STATUS_OK;
}

const SIPLStatus COpenWFDController::SetRect(int sourceWidth, int sourceHeight, uint32_t wfdPipelineIdx) const
{
    WFDint wfdSrcRect[4]{ 0, 0, static_cast<WFDint>(sourceWidth), static_cast<WFDint>(sourceHeight) };
    WFDint wfdDstRect[4]{ 0, 0, m_windowWidths[wfdPipelineIdx], m_windowHeights[wfdPipelineIdx] };

    wfdSetPipelineAttribiv(m_wfdDevice, m_wfdPipelines[wfdPipelineIdx], WFD_PIPELINE_SOURCE_RECTANGLE, 4, wfdSrcRect);
    PGET_WFDERROR_AND_RETURN(m_wfdDevice);

    wfdSetPipelineAttribiv(m_wfdDevice, m_wfdPipelines[wfdPipelineIdx], WFD_PIPELINE_DESTINATION_RECTANGLE, 4,
                           wfdDstRect);
    PGET_WFDERROR_AND_RETURN(m_wfdDevice);

    return NVSIPL_STATUS_OK;
}

const SIPLStatus COpenWFDController::SetEofSyncObj(void) const
{
    return NVSIPL_STATUS_OK;
}

// Streaming functions
const SIPLStatus COpenWFDController::Flip(uint32_t wfdPipelineIdx, uint32_t packetId, NvSciSyncFence *pPostfence)
{
    LOG_DBG("COpenWFDController::Flip, pipeline wfdPipelineIdx = %d, packet id = %u\n", wfdPipelineIdx, packetId);

    wfdBindSourceToPipeline(m_wfdDevice, m_wfdPipelines[wfdPipelineIdx], m_wfdSources[wfdPipelineIdx][packetId],
                            WFD_TRANSITION_AT_VSYNC, NULL);
    PGET_WFDERROR_AND_RETURN(m_wfdDevice);

    wfdDeviceCommitWithNvSciSyncFenceNVX(m_wfdDevice, WFD_COMMIT_PIPELINE, m_wfdPipelines[wfdPipelineIdx], pPostfence);
    PGET_WFDERROR_AND_RETURN(m_wfdDevice);

    return NVSIPL_STATUS_OK;
}
