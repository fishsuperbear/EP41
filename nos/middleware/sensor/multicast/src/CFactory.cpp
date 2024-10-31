// Copyright (c) 2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include "CFactory.hpp"

unique_ptr<CPoolManager> CFactory::CreatePoolManager(uint32_t uSensorId, uint32_t numPackets, bool isC2C)
{
    NvSciStreamBlock poolHandle = 0U;

    auto sciErr = NvSciStreamStaticPoolCreate(numPackets, &poolHandle);
    if (sciErr != NvSciError_Success) {
        LOG_ERR("NvSciStreamStaticPoolCreate failed: 0x%x.\n", sciErr);
        return nullptr;
    }

    return make_unique<CPoolManager>(poolHandle, uSensorId, numPackets, isC2C);
}

void CFactory::GetBasicElementsInfo(vector<ElementInfo> &elemsInfo, vector<uint8_t> &indexs)
{
    indexs.resize(MAX_NUM_ELEMENTS);

    elemsInfo.push_back({ ELEMENT_TYPE_ICP_RAW, false });
    elemsInfo.push_back({ ELEMENT_TYPE_NV12_BL, false });
    elemsInfo.push_back({ ELEMENT_TYPE_METADATA, true });

    indexs[ELEMENT_TYPE_ICP_RAW] = 0U;
    indexs[ELEMENT_TYPE_NV12_BL] = 1U;
    indexs[ELEMENT_TYPE_METADATA] = 2U;

    if (m_pAppConfig->IsMultiElementsEnabled()) {
        elemsInfo.push_back({ ELEMENT_TYPE_NV12_PL, false });
        indexs[ELEMENT_TYPE_NV12_PL] = 3U;
    }
}

void CFactory::GetProducerElementsInfo(ProducerType producerType, vector<ElementInfo> &elemsInfo)
{
    vector<uint8_t> indexs;

    switch (producerType) {
        default:
        case ProducerType_SIPL:
            GetBasicElementsInfo(elemsInfo, indexs);
            elemsInfo[indexs[ELEMENT_TYPE_ICP_RAW]].isUsed = true;
            elemsInfo[indexs[ELEMENT_TYPE_NV12_BL]].isUsed = true;
            if (m_pAppConfig->IsMultiElementsEnabled()) {
                elemsInfo[indexs[ELEMENT_TYPE_NV12_PL]].isUsed = true;
                elemsInfo[indexs[ELEMENT_TYPE_NV12_BL]].hasSibling = true;
                elemsInfo[indexs[ELEMENT_TYPE_NV12_PL]].hasSibling = true;
            }
            break;
        case ProducerType_Display:
            elemsInfo.push_back({ ELEMENT_TYPE_ABGR8888_PL, true });
            break;
    }
}

CProducer *CFactory::CreateProducer(ProducerType producerType, NvSciStreamBlock poolHandle, uint32_t uSensorId)
{
    NvSciStreamBlock producerHandle = 0U;
    vector<ElementInfo> elemsInfo;
    CProducer *pProducer = nullptr;

    auto sciErr = NvSciStreamProducerCreate(poolHandle, &producerHandle);
    if (sciErr != NvSciError_Success) {
        LOG_ERR("NvSciStreamProducerCreate failed: 0x%x.\n", sciErr);
        return nullptr;
    }

    switch (producerType) {
        default:
        case ProducerType_SIPL:
            pProducer = new CSIPLProducer(producerHandle, uSensorId);
            break;
        case ProducerType_Display:
            pProducer = new CDisplayProducer(producerHandle);
            break;
    }

    GetProducerElementsInfo(producerType, elemsInfo);
    pProducer->SetPacketElementsInfo(elemsInfo);

    return pProducer;
}

void CFactory::GetConsumerElementsInfo(ConsumerType consumerType, vector<ElementInfo> &elemsInfo)
{
    vector<uint8_t> indexs;

    switch (consumerType) {
        default:
        case ConsumerType_Enc:
            GetBasicElementsInfo(elemsInfo, indexs);
            elemsInfo[indexs[ELEMENT_TYPE_NV12_BL]].isUsed = true;
            break;
        case ConsumerType_Cuda:
            GetBasicElementsInfo(elemsInfo, indexs);
            if (m_pAppConfig->IsMultiElementsEnabled()) {
                elemsInfo[indexs[ELEMENT_TYPE_NV12_PL]].isUsed = true;
            } else {
                elemsInfo[indexs[ELEMENT_TYPE_NV12_BL]].isUsed = true;
            }
            break;
        case ConsumerType_Stitch:
            GetBasicElementsInfo(elemsInfo, indexs);
            elemsInfo[indexs[ELEMENT_TYPE_NV12_BL]].isUsed = true;
            break;
        case ConsumerType_Display:
            elemsInfo.push_back({ ELEMENT_TYPE_ABGR8888_PL, true });
    }
}

SIPLStatus CFactory::CreateQueue(QueueType queueType, NvSciStreamBlock *pQueueHandle)
{
    NvSciError sciErr = NvSciError_Success;

    if (queueType == QueueType_Mailbox) {
        sciErr = NvSciStreamMailboxQueueCreate(pQueueHandle);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamMailboxQueueCreate");
    } else {
        sciErr = NvSciStreamFifoQueueCreate(pQueueHandle);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamFifoQueueCreate");
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CFactory::CreateConsumerQueueHandles(QueueType queueType,
                                                NvSciStreamBlock *pQueueHandle,
                                                NvSciStreamBlock *pConsumerHandle)
{
    auto status = CreateQueue(queueType, pQueueHandle);
    CHK_STATUS_AND_RETURN(status, "CreateQueue");

    auto sciErr = NvSciStreamConsumerCreate(*pQueueHandle, pConsumerHandle);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamConsumerCreate");

    return NVSIPL_STATUS_OK;
}

unique_ptr<CConsumer> CFactory::CreateConsumer(ConsumerType consumerType, uint32_t uSensorId)
{
    return CreateConsumer(consumerType, uSensorId, m_pAppConfig->GetQueueType());
}

unique_ptr<CConsumer> CFactory::CreateConsumer(ConsumerType consumerType, uint32_t uSensorId, QueueType queueType)
{
    NvSciStreamBlock queueHandle = 0U;
    NvSciStreamBlock consumerHandle = 0U;
    unique_ptr<CConsumer> upCons = nullptr;
    vector<ElementInfo> elemsInfo;

    auto status = CreateConsumerQueueHandles(queueType, &queueHandle, &consumerHandle);
    if (status != NVSIPL_STATUS_OK) {
        return nullptr;
    }

    switch (consumerType) {
        default:
        case ConsumerType_Enc:
            upCons.reset(new CEncConsumer(consumerHandle, uSensorId, queueHandle));
            break;
        case ConsumerType_Cuda:
            upCons.reset(new CCudaConsumer(consumerHandle, uSensorId, queueHandle));
            break;
        case ConsumerType_Stitch:
            upCons.reset(new CStitchingConsumer(consumerHandle, uSensorId, queueHandle));
            break;
        case ConsumerType_Display:
            upCons.reset(new CDisplayConsumer(consumerHandle, uSensorId, queueHandle));
            break;
    }

    upCons->SetAppConfig(m_pAppConfig);

    GetConsumerElementsInfo(consumerType, elemsInfo);
    upCons->SetPacketElementsInfo(elemsInfo);

    return upCons;
}

SIPLStatus CFactory::CreateMulticastBlock(uint32_t consumerCount, NvSciStreamBlock &multicastHandle)
{
    auto sciErr = NvSciStreamMulticastCreate(consumerCount, &multicastHandle);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamMulticastCreate");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CFactory::CreatePresentSync(NvSciSyncModule syncModule, NvSciStreamBlock &presentSyncHandle)
{
    NvSciError sciErr = NvSciStreamPresentSyncCreate(syncModule, &presentSyncHandle);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamPresentSyncCreate");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CFactory::OpenEndpoint(const char *channel, NvSciIpcEndpoint *pEndPoint)
{
    /* Open the named channel */
    auto sciErr = NvSciIpcOpenEndpoint(channel, pEndPoint);
    std::cout << "channel: " << channel << std::endl;
    if (NvSciError_Success != sciErr) {
        LOG_ERR("Failed (0x%x) to open channel (%s)\n", sciErr, channel);
        return NVSIPL_STATUS_ERROR;
    }
    (void)NvSciIpcResetEndpointSafe(*pEndPoint);

    return NVSIPL_STATUS_OK;
}

void CFactory::CloseEndpoint(NvSciIpcEndpoint &endPoint)
{
    if (endPoint) {
        (void)NvSciIpcCloseEndpointSafe(endPoint, false);
    }
}

SIPLStatus CFactory::CreateIpcBlock(NvSciSyncModule syncModule,
                                    NvSciBufModule bufModule,
                                    const char *channel,
                                    bool isSrc,
                                    NvSciIpcEndpoint *pEndPoint,
                                    NvSciStreamBlock *pIpcBlock)
{
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

SIPLStatus CFactory::ReleaseIpcBlock(NvSciIpcEndpoint pEndpoint, NvSciStreamBlock pIpcBlock)
{
    if (!pIpcBlock || !pEndpoint) {
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }

    (void)NvSciStreamBlockDelete(pIpcBlock);
    NvSciIpcCloseEndpoint(pEndpoint);
    return NVSIPL_STATUS_OK;
}

SIPLStatus CFactory::CreateC2CSrc(NvSciSyncModule syncModule,
                                  NvSciBufModule bufModule,
                                  const char *channel,
                                  NvSciStreamBlock c2cQueueHandle,
                                  NvSciIpcEndpoint *pEndPoint,
                                  NvSciStreamBlock *pIpcBlock)
{
    auto status = OpenEndpoint(channel, pEndPoint);
    CHK_STATUS_AND_RETURN(status, "OpenEndpoint");

    auto sciErr = NvSciStreamIpcSrcCreate2(*pEndPoint, syncModule, bufModule, c2cQueueHandle, pIpcBlock);
    if (sciErr != NvSciError_Success) {
        CloseEndpoint(*pEndPoint);
        LOG_ERR("NvSciStreamIpcSrcCreate2 failed, status: 0x%x\n", sciErr);
        return NVSIPL_STATUS_ERROR;
    }

    return NVSIPL_STATUS_OK;
}

SIPLStatus CFactory::CreateC2CDst(NvSciSyncModule syncModule,
                                  NvSciBufModule bufModule,
                                  const char *channel,
                                  NvSciStreamBlock poolHandle,
                                  NvSciIpcEndpoint *pEndPoint,
                                  NvSciStreamBlock *pIpcBlock)
{
    auto status = OpenEndpoint(channel, pEndPoint);
    CHK_STATUS_AND_RETURN(status, "OpenEndpoint");

    auto sciErr = NvSciStreamIpcDstCreate2(*pEndPoint, syncModule, bufModule, poolHandle, pIpcBlock);
    if (sciErr != NvSciError_Success) {
        CloseEndpoint(*pEndPoint);
        LOG_ERR("NvSciStreamIpcDstCreate2 failed, status: 0x%x\n", sciErr);
        return NVSIPL_STATUS_ERROR;
    }

    return NVSIPL_STATUS_OK;
}
