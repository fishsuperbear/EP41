// Copyright (c) 2022-2023 NVIDIA Corporation. All rights reserved.
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
#include "CStitchingConsumer.hpp"
#include "CDisplayConsumer.hpp"
#include "CDisplayProducer.hpp"
#include "CLateConsumerHelper.hpp"

#include "nvscibuf.h"

using namespace std;
using namespace nvsipl;

class CFactory
{
  public:
    static CFactory &GetInstance(CAppConfig *pAppConfig)
    {
        static CFactory instance(pAppConfig);
        return instance;
    }

    unique_ptr<CPoolManager> CreatePoolManager(uint32_t uSensorId, uint32_t numPackets, bool isC2C = false);

    CProducer *CreateProducer(ProducerType producerType, NvSciStreamBlock poolHandle, uint32_t uSensorId = 0U);

    SIPLStatus CreateQueue(QueueType queueType, NvSciStreamBlock *pQueueHandle);

    SIPLStatus
    CreateConsumerQueueHandles(QueueType queueType, NvSciStreamBlock *pQueueHandle, NvSciStreamBlock *pConsumerHandle);

    unique_ptr<CConsumer> CreateConsumer(ConsumerType consumerType, uint32_t uSensorId);
    unique_ptr<CConsumer> CreateConsumer(ConsumerType consumerType, uint32_t uSensorId, QueueType queueType);

    SIPLStatus CreateMulticastBlock(uint32_t consumerCount, NvSciStreamBlock &multicastHandle);

    SIPLStatus CreatePresentSync(NvSciSyncModule syncModule, NvSciStreamBlock &presentSyncHandle);

    SIPLStatus OpenEndpoint(const char *channel, NvSciIpcEndpoint *pEndPoint);

    SIPLStatus CreateIpcBlock(NvSciSyncModule syncModule,
                              NvSciBufModule bufModule,
                              const char *channel,
                              bool isSrc,
                              NvSciIpcEndpoint *pEndPoint,
                              NvSciStreamBlock *pIpcBlock);

    SIPLStatus ReleaseIpcBlock(NvSciIpcEndpoint pEndpoint, NvSciStreamBlock pIpcBlock);

    SIPLStatus CreateC2CSrc(NvSciSyncModule syncModule,
                            NvSciBufModule bufModule,
                            const char *channel,
                            NvSciStreamBlock c2cQueueHandle,
                            NvSciIpcEndpoint *pEndPoint,
                            NvSciStreamBlock *pIpcBlock);

    SIPLStatus CreateC2CDst(NvSciSyncModule syncModule,
                            NvSciBufModule bufModule,
                            const char *channel,
                            NvSciStreamBlock poolHandle,
                            NvSciIpcEndpoint *pEndPoint,
                            NvSciStreamBlock *pIpcBlock);

  private:
    CFactory(CAppConfig *pAppConfig)
        : m_pAppConfig(pAppConfig)
    {
    }

    CFactory(const CFactory &obj) = delete;
    CFactory &operator=(const CFactory &obj) = delete;

    void GetBasicElementsInfo(vector<ElementInfo> &elemsInfo, vector<uint8_t> &indexs);
    void GetProducerElementsInfo(ProducerType producerType, vector<ElementInfo> &elemsInfo);
    void GetConsumerElementsInfo(ConsumerType consumerType, vector<ElementInfo> &elemsInfo);
    void CloseEndpoint(NvSciIpcEndpoint &endPoint);

    CAppConfig *m_pAppConfig = nullptr;
};

#endif
