// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef CFACTORY_H
#define CFACTORY_H

#include "CUtils.hpp"
#include "CPoolManager.hpp"
#include "CSIPLProducer.hpp"

#include "nvscibuf.h"
#include <cstdint>

using namespace std;
using namespace nvsipl;

class CFactory
{
public:
    CFactory() {}
    ~CFactory();

    static std::unique_ptr<CPoolManager> CreatePoolManager(uint32_t uSensor, uint32_t numPackets)
    {
        NvSciStreamBlock poolHandle = 0U;
        auto sciErr = NvSciStreamStaticPoolCreate(numPackets, &poolHandle);
        if (sciErr != NvSciError_Success) {
            LOG_ERR("NvSciStreamStaticPoolCreate failed: 0x%x.\n", sciErr);
            return nullptr;
        }
        return std::unique_ptr<CPoolManager>(new CPoolManager(poolHandle, uSensor, numPackets, false));
    }

    static std::unique_ptr<CProducer> CreateProducer(NvSciStreamBlock poolHandle, const SensorInfo* pSensorInfo)
    {
        NvSciStreamBlock producerHandle = 0U;
        vector<ElementInfo> elemsInfo;
        unique_ptr<CProducer> upProducer = nullptr;

        auto sciErr = NvSciStreamProducerCreate(poolHandle, &producerHandle);
        if (sciErr != NvSciError_Success) {
            LOG_ERR("NvSciStreamProducerCreate failed: 0x%x.\n", sciErr);
            return nullptr;
        }
        upProducer.reset(new CSIPLProducer(producerHandle, pSensorInfo->id));

        GetProducerElementsInfo(pSensorInfo->id,elemsInfo);
        upProducer->SetPacketElementsInfo(elemsInfo);

        return upProducer;
    }

    static SIPLStatus CreateMulticastBlock(uint32_t consumerCount, NvSciStreamBlock& multicastHandle)
    {
        auto sciErr = NvSciStreamMulticastCreate(consumerCount, &multicastHandle);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamMulticastCreate");

        return NVSIPL_STATUS_OK;
    }

    static SIPLStatus OpenEndpoint(const char *channel, NvSciIpcEndpoint *pEndPoint){
        /* Open the named channel */
        auto sciErr = NvSciIpcOpenEndpoint(channel, pEndPoint);
        if (NvSciError_Success != sciErr) {
            LOG_ERR("Failed (0x%x) to open channel (%s)\n", sciErr, channel);
            return NVSIPL_STATUS_ERROR;
        }
        (void)NvSciIpcResetEndpointSafe(*pEndPoint);

        return NVSIPL_STATUS_OK;
    }

    static void CloseEndpoint(NvSciIpcEndpoint &endPoint)
    {
        if (endPoint) {
            (void)NvSciIpcCloseEndpointSafe(endPoint, false);
        }
    }

    static SIPLStatus CreateLimiterBlock(uint32_t limiterCount, NvSciStreamBlock& limiterHandle)
    {
        auto sciErr = NvSciStreamLimiterCreate(limiterCount, &limiterHandle);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "CreateLimiterBlock");

        return NVSIPL_STATUS_OK;
    }

    static SIPLStatus CreateReturnSyncBlock(NvSciSyncModule syncModule, NvSciStreamBlock& syncHandle)
    {
        auto sciErr = NvSciStreamReturnSyncCreate(syncModule, &syncHandle);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "CreateLimiterBlock");

        return NVSIPL_STATUS_OK;
    }

    static SIPLStatus CreateIpcBlock(NvSciSyncModule syncModule, NvSciBufModule bufModule,
                                    const char *channel, bool isSrc,
                                    NvSciIpcEndpoint *pEndPoint, NvSciStreamBlock *pIpcBlock){
        auto status = OpenEndpoint(channel, pEndPoint);
        CHK_STATUS_AND_RETURN(status, "OpenEndpoint");

        /* Create an ipc block */
        auto sciErr = isSrc ? NvSciStreamIpcSrcCreate(*pEndPoint, syncModule, bufModule, pIpcBlock)
                            : NvSciStreamIpcDstCreate(*pEndPoint, syncModule, bufModule, pIpcBlock);
        if (sciErr != NvSciError_Success) {
            CloseEndpoint(*pEndPoint);
            LOG_ERR("Create ipc block failed, status: 0x%x, isSrc: %u\n", sciErr, isSrc);
            return NVSIPL_STATUS_ERROR;
        }

        return NVSIPL_STATUS_OK;
    }

    static SIPLStatus CreateIpcBlock(NvSciSyncModule syncModule, NvSciBufModule bufModule,
                                           const char* channel, bool isSrc, NvSciStreamBlock* ipcBlock)
    {
        NvSciIpcEndpoint  endpoint;
        NvSciStreamBlock  block = 0U;

        /* Open the named channel */
        auto sciErr = NvSciIpcOpenEndpoint(channel, &endpoint);
        if (NvSciError_Success != sciErr) {
            LOG_ERR("Failed (0x%x) to open channel (%s) for IpcSrc\n", sciErr, channel);
            return NVSIPL_STATUS_ERROR;
        }
        NvSciIpcResetEndpoint(endpoint);

        /* Create an ipc block */
        if (isSrc) {
            sciErr = NvSciStreamIpcSrcCreate(endpoint, syncModule, bufModule, &block);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamIpcSrcCreate");
        } else {
            sciErr = NvSciStreamIpcDstCreate(endpoint, syncModule, bufModule, &block);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamIpcDstCreate");
        }

        *ipcBlock = block;
        return NVSIPL_STATUS_OK;
    }

    static SIPLStatus ReleaseIpcBlock(NvSciIpcEndpoint pEndpoint, NvSciStreamBlock pIpcBlock)
    {
        if (!pIpcBlock || !pEndpoint) {
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }

        (void)NvSciStreamBlockDelete(pIpcBlock);
        NvSciIpcCloseEndpoint(pEndpoint);
        return NVSIPL_STATUS_OK;
    }

    static void GetBasicElementsInfo(vector<ElementInfo> &elemsInfo, vector<uint8_t> &indexs)
    {
        indexs.resize(MAX_NUM_ELEMENTS);

        elemsInfo.push_back({ ELEMENT_TYPE_ICP_RAW, false });
        elemsInfo.push_back({ ELEMENT_TYPE_NV12_BL, false });
        elemsInfo.push_back({ ELEMENT_TYPE_METADATA, true });

        indexs[ELEMENT_TYPE_ICP_RAW] = 0U;
        indexs[ELEMENT_TYPE_NV12_BL] = 1U;
        indexs[ELEMENT_TYPE_METADATA] = 2U;

        // if (m_pAppConfig->IsMultiElementsEnabled()) {
        //     elemsInfo.push_back({ ELEMENT_TYPE_NV12_PL, false });
        //     indexs[ELEMENT_TYPE_NV12_PL] = 3U;
        // }
    }

    static void GetProducerElementsInfo(uint32_t uSensor, vector<ElementInfo>& elemsInfo) {
        vector<uint8_t> indexs;
        const NVSensorConfig*  nvsensorConfig= NVPlatformConfig::getInstance().getSensorConfig(uSensor);

        GetBasicElementsInfo(elemsInfo, indexs);
        elemsInfo[indexs[ELEMENT_TYPE_ICP_RAW]].isUsed = true;
        if (nvsensorConfig->enableISP0) {
            elemsInfo[indexs[ELEMENT_TYPE_NV12_BL]].isUsed = true;
        }

        // if (m_pAppConfig->IsMultiElementsEnabled()) {
        //     elemsInfo[indexs[ELEMENT_TYPE_NV12_PL]].isUsed = true;
        //     elemsInfo[indexs[ELEMENT_TYPE_NV12_BL]].hasSibling = true;
        //     elemsInfo[indexs[ELEMENT_TYPE_NV12_PL]].hasSibling = true;
        // }
    }

};

#endif
