// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef CFACTORY_H
#define CFACTORY_H

#include "sensor/nvs_consumer/CUtils.hpp"
#include "sensor/nvs_consumer/CEncConsumer.hpp"
#include "sensor/nvs_consumer/CCudaConsumer.hpp"
#include "sensor/nvs_consumer/CNvMediaConsumer.hpp"
#include "sensor/nvs_consumer/CDisplayConsumer.hpp"
#include "nvscibuf.h"

namespace hozon {
namespace netaos {
namespace desay { 

using namespace std;
using namespace nvsipl;

class CFactory
{
public:
    CFactory() {}
    ~CFactory();
    
    static std::unique_ptr<CConsumer> CreateConsumer(ConsumerType consumerType, SensorInfo *pSensorInfo, bool bUseMailbox)
    {
        NvSciStreamBlock queueHandle = 0U;
        NvSciStreamBlock consumerHandle = 0U;
        NvSciError sciErr = NvSciError_Success;
        vector<ElementInfo> elemsInfo;
        unique_ptr<CConsumer> upCons = nullptr;

        if (bUseMailbox) {
            sciErr = NvSciStreamMailboxQueueCreate(&queueHandle);
        } else {
            sciErr = NvSciStreamFifoQueueCreate(&queueHandle);
        }
        if (sciErr != NvSciError_Success) {
            LOG_ERR("NvSciStreamFifoQueueCreate failed: 0x%x.\n", sciErr);
            return nullptr;
        }
        sciErr = NvSciStreamConsumerCreate(queueHandle, &consumerHandle);
        if (sciErr != NvSciError_Success) {
            LOG_ERR("NvSciStreamConsumerCreate failed: 0x%x.\n", sciErr);
            return nullptr;
        }

        if (consumerType == CUDA_CONSUMER) {
            upCons.reset(new CCudaConsumer(consumerHandle, pSensorInfo->id, queueHandle));
        } else if(consumerType == ENC_CONSUMER) {
            auto encodeWidth = (uint16_t)pSensorInfo->vcInfo.resolution.width;
            auto encodeHeight = (uint16_t)pSensorInfo->vcInfo.resolution.height;

            upCons.reset(new CEncConsumer(consumerHandle, pSensorInfo->id, queueHandle, encodeWidth, encodeHeight));
        // }else{
        //     auto encodeWidth = (uint16_t)pSensorInfo->vcInfo.resolution.width;
        //     auto encodeHeight = (uint16_t)pSensorInfo->vcInfo.resolution.height;
        //     return std::unique_ptr<CNvMediaConsumer>(new CNvMediaConsumer(consumerHandle, pSensorInfo->id, queueHandle, encodeWidth, encodeHeight));
        } else if (consumerType == DISPLAY_CONSUMER) {
            upCons.reset(new CDisplayConsumer(consumerHandle, pSensorInfo->id, queueHandle));
        }
        GetConsumerElementsInfo(pSensorInfo->id, consumerType, elemsInfo);
        upCons->SetPacketElementsInfo(elemsInfo);

        return upCons;
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

    static void GetConsumerElementsInfo(uint32_t uSensor, ConsumerType consumerType, vector<ElementInfo> &elemsInfo)
    {
        vector<uint8_t> indexs;

        switch (consumerType) {
            default:
            case ENC_CONSUMER:
                GetBasicElementsInfo(elemsInfo, indexs);
                if(uSensor<2){
                    elemsInfo[indexs[ELEMENT_TYPE_NV12_BL]].isUsed = true;
                }else{
                    elemsInfo[indexs[ELEMENT_TYPE_ICP_RAW]].isUsed = true;
                }

                break;
            case CUDA_CONSUMER:
                GetBasicElementsInfo(elemsInfo, indexs);
                // if (m_pAppConfig->IsMultiElementsEnabled()) {
                //     elemsInfo[indexs[ELEMENT_TYPE_NV12_PL]].isUsed = true;
                // } else {
                if(uSensor<2){
                    elemsInfo[indexs[ELEMENT_TYPE_NV12_BL]].isUsed = true;
                }else{
                    elemsInfo[indexs[ELEMENT_TYPE_ICP_RAW]].isUsed = true;
                }
                    
                // }
                break;
            case DISPLAY_CONSUMER:
                GetBasicElementsInfo(elemsInfo, indexs);
                if(uSensor<2){
                    elemsInfo[indexs[ELEMENT_TYPE_NV12_BL]].isUsed = true;
                }else{
                    elemsInfo[indexs[ELEMENT_TYPE_ICP_RAW]].isUsed = true;
                }
                break;
        }
    }

};

}
}
}

#endif
