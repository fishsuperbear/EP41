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
#include "CCudaConsumer.hpp"
#include "CEncConsumer.hpp"
#include "CCommonConsumer.hpp"

#include "nvscibuf.h"

using namespace std;
using namespace nvsipl;

class CFactory
{
public:
    CFactory() {}
    ~CFactory();

    static std::unique_ptr<CPoolManager> CreatePoolManager(uint32_t uSensor)
    {
        NvSciStreamBlock poolHandle = 0U;
        auto sciErr = NvSciStreamStaticPoolCreate(MAX_NUM_PACKETS, &poolHandle);
        if (sciErr != NvSciError_Success) {
            LOG_ERR("NvSciStreamStaticPoolCreate failed: 0x%x.\n", sciErr);
            return nullptr;
        }
        return std::unique_ptr<CPoolManager>(new CPoolManager(poolHandle, uSensor));
    }

    static std::unique_ptr<CProducer> CreateProducer(NvSciStreamBlock poolHandle, uint32_t uSensor, INvSIPLCamera* pCamera)
    {
        NvSciStreamBlock producerHandle = 0U;

        auto sciErr = NvSciStreamProducerCreate(poolHandle, &producerHandle);
        if (sciErr != NvSciError_Success) {
            LOG_ERR("NvSciStreamProducerCreate failed: 0x%x.\n", sciErr);
            return nullptr;
        }
        return std::unique_ptr<CProducer>(new CSIPLProducer(producerHandle, uSensor, pCamera));
    }

    static std::unique_ptr<CConsumer> CreateConsumer(ConsumerType consumerType, SensorInfo *pSensorInfo, bool bUseMailbox)
    {
        NvSciStreamBlock queueHandle = 0U;
        NvSciStreamBlock consumerHandle = 0U;
        NvSciError sciErr = NvSciError_Success;

        if (bUseMailbox) {
            sciErr = NvSciStreamMailboxQueueCreate(&queueHandle);
            if (sciErr != NvSciError_Success) {
                LOG_ERR("NvSciStreamMailboxQueueCreate failed: 0x%x.\n", sciErr);
                return nullptr;
            }
        } else {
            sciErr = NvSciStreamFifoQueueCreate(&queueHandle);
            if (sciErr != NvSciError_Success) {
                LOG_ERR("NvSciStreamFifoQueueCreate failed: 0x%x.\n", sciErr);
                return nullptr;
            }
        }

        sciErr = NvSciStreamConsumerCreate(queueHandle, &consumerHandle);
        if (sciErr != NvSciError_Success) {
            LOG_ERR("NvSciStreamConsumerCreate failed: 0x%x.\n", sciErr);
            return nullptr;
        }

        if (consumerType == CUDA_CONSUMER) {
            return std::unique_ptr<CConsumer>(new CCudaConsumer(consumerHandle, pSensorInfo->id, queueHandle));
        } else if (consumerType == ENC_CONSUMER){
            auto encodeWidth = (uint16_t)pSensorInfo->vcInfo.resolution.width;
            auto encodeHeight = (uint16_t)pSensorInfo->vcInfo.resolution.height;

            return std::unique_ptr<CConsumer>(new CEncConsumer(consumerHandle, pSensorInfo->id, queueHandle, encodeWidth, encodeHeight));
        }
        else {
            return std::unique_ptr<CConsumer>(new CCommonConsumer(consumerHandle, pSensorInfo->id, queueHandle));
        }
    }

    static SIPLStatus CreateMulticastBlock(uint32_t consumerCount, NvSciStreamBlock& multicastHandle)
    {
        auto sciErr = NvSciStreamMulticastCreate(consumerCount, &multicastHandle);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamMulticastCreate");

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
            LOG_ERR("Failed (0x%x) to open channel (%s)\n", sciErr, channel);
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
};

#endif
