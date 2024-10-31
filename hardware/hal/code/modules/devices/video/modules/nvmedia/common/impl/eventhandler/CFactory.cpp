#include "CFactory.hpp"

std::unique_ptr<CPoolManager> CFactory::CreatePoolManager(uint32_t uSensor, uint32_t numPackets, std::shared_ptr<CAttributeProvider> attrProvider)
{
    NvSciStreamBlock poolHandle = 0U;
    auto sciErr = NvSciStreamStaticPoolCreate( numPackets, &poolHandle);
    if (sciErr != NvSciError_Success) {
        LOG_ERR("NvSciStreamStaticPoolCreate failed: 0x%x.\r\n", sciErr);
        return nullptr;
    }
    return std::unique_ptr<CPoolManager>(new CPoolManager(poolHandle, uSensor, numPackets,attrProvider));
}

std::vector<ElementInfo> CFactory::GetElementsInfo(UseCaseInfo &ucInfo)
    {
        std::vector<ElementInfo> elemsInfo;
        elemsInfo.resize(MAX_NUM_ELEMENTS);
        if(ucInfo.isEnableICP){
            elemsInfo[ELEMENT_TYPE_ICP_RAW] = { ELEMENT_TYPE_ICP_RAW, true };
            elemsInfo[ELEMENT_TYPE_NV12_BL] = { ELEMENT_TYPE_NV12_BL, false };
        }else{
            elemsInfo[ELEMENT_TYPE_ICP_RAW] = { ELEMENT_TYPE_ICP_RAW, false };
            elemsInfo[ELEMENT_TYPE_NV12_BL] = { ELEMENT_TYPE_NV12_BL, true };
        }
        elemsInfo[ELEMENT_TYPE_METADATA] = { ELEMENT_TYPE_METADATA, true };

        if (ucInfo.isMultiElems) {
            elemsInfo[ELEMENT_TYPE_NV12_PL] = { ELEMENT_TYPE_NV12_PL, true };
        }

        return elemsInfo;
    }

std::unique_ptr<CProducer> CFactory::CreateVICProducer(NvSciStreamBlock poolHandle, uint32_t uSensor, ICascadedProvider* pCascadedProvider, UseCaseInfo &ucInfo) {
    NvSciStreamBlock producerHandle = 0U;

    auto sciErr = NvSciStreamProducerCreate(poolHandle, &producerHandle);
    if (sciErr != NvSciError_Success)
    {
        LOG_ERR("NvSciStreamProducerCreate failed: 0x%x.\n", sciErr);
        return nullptr;
    }

    std::unique_ptr<CProducer> prod(new CVICProducer(producerHandle, uSensor, pCascadedProvider));
    std::vector<ElementInfo> elemsInfo = GetElementsInfo(ucInfo);
    /* elemsInfo[ELEMENT_TYPE_ICP_RAW].isUsed = true; */
    //if (ucInfo.isMultiElems) {
    //    elemsInfo[ELEMENT_TYPE_NV12_BL].hasSibling = true;
    //    elemsInfo[ELEMENT_TYPE_NV12_PL].hasSibling = true;
    //}
    prod->SetPacketElementsInfo(elemsInfo);

    return prod;
}


std::unique_ptr<CProducer> CFactory::CreateProducer(NvSciStreamBlock poolHandle, uint32_t uSensor, uint32_t i_outputtype, INvSIPLCamera* pCamera, UseCaseInfo &ucInfo, std::shared_ptr<CAttributeProvider> attrProvider)
{
    NvSciStreamBlock producerHandle = 0U;

    auto sciErr = NvSciStreamProducerCreate(poolHandle, &producerHandle);
    if (sciErr != NvSciError_Success) {
        LOG_ERR("NvSciStreamProducerCreate failed: 0x%x.\r\n", sciErr);
        return nullptr;
    }
    std::unique_ptr<CProducer> prod(new CSIPLProducer(producerHandle, uSensor, i_outputtype, pCamera,attrProvider));
    std::vector<ElementInfo> elemsInfo = GetElementsInfo(ucInfo);
    elemsInfo[ELEMENT_TYPE_ICP_RAW].isUsed = true;
    if (ucInfo.isMultiElems) {
        elemsInfo[ELEMENT_TYPE_NV12_BL].hasSibling = true;
        elemsInfo[ELEMENT_TYPE_NV12_PL].hasSibling = true;
    }
    prod->SetPacketElementsInfo(elemsInfo);

    return prod;
}

std::unique_ptr<CProducer> CFactory::CreateIPCProducer(NvSciStreamBlock poolHandle,
                                                       uint32_t uSensor,
                                                       uint32_t i_outputtype,
                                                       INvSIPLCamera* pCamera,
                                                       UseCaseInfo &ucInfo, std::shared_ptr<CAttributeProvider> attrProvider)
{
    NvSciStreamBlock producerHandle = 0U;

    auto sciErr = NvSciStreamProducerCreate(poolHandle, &producerHandle);
    if (sciErr != NvSciError_Success) {
        LOG_ERR("NvSciStreamProducerCreate failed: 0x%x.\r\n", sciErr);
        return nullptr;
    }

    std::unique_ptr<CProducer> prod(new CSIPLProducer(producerHandle, uSensor, i_outputtype, pCamera,attrProvider));

    std::vector<ElementInfo> elemsInfo = GetElementsInfo(ucInfo);
    elemsInfo[ELEMENT_TYPE_ICP_RAW].isUsed = true;
    prod->SetPacketElementsInfo(elemsInfo);

    return prod;
}

std::unique_ptr<CConsumer> CFactory::CreateConsumer(ConsumerType consumerType, SensorInfo* pSensorInfo, uint32_t i_outputtype, bool bUseMailbox, UseCaseInfo& ucInfo, HWNvmediaOutputPipelineContext* i_poutputpipeline, int encodeType, void* i_pvicconsumer) {
    NvSciStreamBlock queueHandle = 0U;
    NvSciStreamBlock consumerHandle = 0U;
    NvSciError sciErr = NvSciError_Success;


    if (bUseMailbox) {
        sciErr = NvSciStreamMailboxQueueCreate(&queueHandle);
        if (sciErr != NvSciError_Success) {
            LOG_ERR("NvSciStreamMailboxQueueCreate failed: 0x%x.\r\n", sciErr);
            return nullptr;
        }
    }
    else {
        sciErr = NvSciStreamFifoQueueCreate(&queueHandle);
        if (sciErr != NvSciError_Success) {
            LOG_ERR("NvSciStreamFifoQueueCreate failed: 0x%x.\r\n", sciErr);
            return nullptr;
        }
    }

    sciErr = NvSciStreamConsumerCreate(queueHandle, &consumerHandle);
    if (sciErr != NvSciError_Success) {
        LOG_ERR("NvSciStreamConsumerCreate failed: 0x%x.\r\n", sciErr);
        return nullptr;
    }

    u32 capturewidth, captureheight;
    capturewidth = (u32)pSensorInfo->vcInfo.resolution.width;
    captureheight = (u32)pSensorInfo->vcInfo.resolution.height;
    std::unique_ptr<CConsumer> upCons;
    std::vector<ElementInfo> elemsInfo = GetElementsInfo(ucInfo);
    u32 blockindex = i_poutputpipeline->poutputpipeline_ops->blockindex;
    u32 sensorindex = i_poutputpipeline->poutputpipeline_ops->sensorindex;

    /*
    * The following consumer input parameter(pSensorInfo->id£¬ blockindex and sensorindex) is for trace only.
    * We add the logic blockindex and logic sensorindex.
    */
    if (consumerType == CUDA_CONSUMER) {
        upCons.reset(new CCudaConsumer(consumerHandle, pSensorInfo->id, blockindex, sensorindex, queueHandle, capturewidth, captureheight));
        if (ucInfo.isMultiElems) {
                elemsInfo[ELEMENT_TYPE_NV12_BL].isUsed = false;
        }
    }
    else if (consumerType == ENC_CONSUMER) {
        auto encodeWidth = (uint16_t)pSensorInfo->vcInfo.resolution.width;
        auto encodeHeight = (uint16_t)pSensorInfo->vcInfo.resolution.height;
        upCons.reset(new CEncConsumer(consumerHandle, pSensorInfo->id, blockindex, sensorindex, queueHandle, encodeWidth, encodeHeight, encodeType, i_pvicconsumer));
        if (ucInfo.isMultiElems) {
            elemsInfo[ELEMENT_TYPE_NV12_PL].isUsed = false;
        }
    }
    else if (consumerType == COMMON_CONSUMER) {
        upCons.reset(new CCommonConsumer(consumerHandle, pSensorInfo->id, i_outputtype, queueHandle, blockindex, sensorindex));
        if (ucInfo.isMultiElems) {
            elemsInfo[ELEMENT_TYPE_NV12_PL].isUsed = false;
        }
    }
    else if (consumerType == VIC_CONSUMER) {
        upCons.reset(new CVICConsumer(consumerHandle, pSensorInfo, queueHandle, i_poutputpipeline,encodeType));
        if (ucInfo.isMultiElems) {
            elemsInfo[ELEMENT_TYPE_NV12_PL].isUsed = false;
        }
    }
    upCons->SetPacketElementsInfo(elemsInfo);

        return upCons;
}

SIPLStatus CFactory::CreateMulticastBlock(uint32_t consumerCount, NvSciStreamBlock& multicastHandle)
{
    auto sciErr = NvSciStreamMulticastCreate(consumerCount, &multicastHandle);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamMulticastCreate");

    return NVSIPL_STATUS_OK;
}

SIPLStatus CFactory::CreateIpcBlock(NvSciSyncModule syncModule, NvSciBufModule bufModule,
    const char* channel, bool isSrc, NvSciStreamBlock* ipcBlock, NvSciIpcEndpoint *ipcEndpoint)
{
    NvSciIpcEndpoint  endpoint;
    NvSciStreamBlock  block = 0U;

    /* Open the named channel */
    auto sciErr = NvSciIpcOpenEndpoint(channel, &endpoint);
    if (NvSciError_Success != sciErr) {
        LOG_ERR("Failed (0x%x) to open channel (%s)\r\n", sciErr, channel);
        return NVSIPL_STATUS_ERROR;
    }
    NvSciIpcResetEndpointSafe(endpoint);

    /* Create an ipc block */
    if (isSrc) {
        sciErr = NvSciStreamIpcSrcCreate(endpoint, syncModule, bufModule, &block);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamIpcSrcCreate");
    }
    else {
        sciErr = NvSciStreamIpcDstCreate(endpoint, syncModule, bufModule, &block);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamIpcDstCreate");
    }

    *ipcBlock = block;
    if (ipcEndpoint != nullptr)
    {
        *ipcEndpoint = endpoint;
    }
    return NVSIPL_STATUS_OK;
}

SIPLStatus CFactory::ReleaseIpcBlock(NvSciStreamBlock ipcBlock, NvSciIpcEndpoint ipcEndpoint)
{
    if (!ipcBlock || !ipcEndpoint) {
        return NVSIPL_STATUS_BAD_ARGUMENT;
    }


    (void)NvSciStreamBlockDelete(ipcBlock);
    NvSciIpcCloseEndpoint(ipcEndpoint);
    return NVSIPL_STATUS_OK;
}
