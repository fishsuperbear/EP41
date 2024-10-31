#include "hw_nvmedia_common_impl.h"
#include "hw_nvmedia_eventhandler_impl.h"
#define NVMEIDA_PIPELIN_VALID 1
#define NVMEIDA_PIPELIN_INVALID 0

hw_ret_s32 HWNvmediaSingleProcessContext::Pipeline_Open(struct hw_video_blockspipelineconfig_t* i_pblockspipelineconfig,
	struct hw_video_blockspipeline_ops_t** o_ppblockspipeline_ops)
{
	HW_NVMEDIA_LOG_UNMASK("Pipeline Open Enter!\r\n");
	CHK_LOG_SENTENCE_HW_RET_S32(pipelineopen(i_pblockspipelineconfig));
    CHK_LOG_SENTENCE_HW_RET_S32(pipelineprepare());
    *o_ppblockspipeline_ops = _pblockspipeline_ops;

	return 0;
}

hw_ret_s32 HWNvmediaSingleProcessContext::Pipeline_Close()
{
    HW_NVMEDIA_LOG_UNMASK("Pipeline Close Enter!\r\n");
    CHK_LOG_SENTENCE_HW_RET_S32(pipelineunprepare());
    CHK_LOG_SENTENCE_HW_RET_S32(pipelineclose());

	return 0;
}

hw_ret_s32 HWNvmediaSingleProcessContext::Pipeline_Start()
{
    HW_NVMEDIA_LOG_UNMASK("Pipeline Start Enter!\r\n");
    CHK_LOG_SENTENCE_HW_RET_S32(pipelinestart());
	return 0;
}

hw_ret_s32 HWNvmediaSingleProcessContext::Pipeline_Stop()
{
    HW_NVMEDIA_LOG_UNMASK("Pipeline Stop Enter!\r\n");
    CHK_LOG_SENTENCE_HW_RET_S32(pipelinestop());
	return 0;
}

hw_ret_s32 HWNvmediaSingleProcessContext::CreateBlocks(HWNvmediaOutputPipelineContext* io_poutputpipeline)
{
    HW_NVMEDIA_LOG_DEBUG("CreateBlocks.\r\n");
    uint8_t consumer_num=0;
    int encodeType = HW_VIDEO_REGDATACB_TYPE_AVC;
    UseCaseInfo ucInfo;

    hw_video_sensorpipelineconfig_t* psensorpipelineconfig =
        &_blockspipelineconfig.parrayblock[io_poutputpipeline->poutputpipeline_ops->blockindex].parraysensor[io_poutputpipeline->poutputpipeline_ops->sensorindex];
    for(u32 i=0;i<psensorpipelineconfig->datacbsconfig.arraynumdatacbs;i++){
        if(psensorpipelineconfig->datacbsconfig.parraydatacbs[i].type==HW_VIDEO_REGDATACB_TYPE_HEVC){
            encodeType = HW_VIDEO_REGDATACB_TYPE_HEVC;
        }else if(psensorpipelineconfig->datacbsconfig.parraydatacbs[i].type==HW_VIDEO_REGDATACB_TYPE_AVC){
            encodeType = HW_VIDEO_REGDATACB_TYPE_AVC;
        }
    }
    if(psensorpipelineconfig->bcaptureoutputrequested){// enable ICP
        ucInfo.isMultiElems = false;
        ucInfo.isEnableICP = true;
    }else if(psensorpipelineconfig->bisp0outputrequested && psensorpipelineconfig->bisp1outputrequested){//enable ISP0 and ISP1
        ucInfo.isMultiElems = true;
    }else if(psensorpipelineconfig->bisp0outputrequested && !psensorpipelineconfig->bisp1outputrequested){//enable ISP0
        ucInfo.isMultiElems = false;
    }else if(!psensorpipelineconfig->bisp0outputrequested && psensorpipelineconfig->bisp1outputrequested){//enable ISP1
        ucInfo.isMultiElems = false;
    }

    io_poutputpipeline->_peventhandler->ppoolmanager = CFactory::CreatePoolManager(io_poutputpipeline->psensorinfo->id, MAX_NUM_PACKETS);
    CHK_PTR_AND_RET_S32(io_poutputpipeline->_peventhandler->ppoolmanager, "CFactory::CreatePoolManager.");
    HW_NVMEDIA_LOG_DEBUG("PoolManager is created.\r\n");

    std::unique_ptr<CProducer> upProducer = CFactory::CreateProducer(io_poutputpipeline->_peventhandler->ppoolmanager->GetHandle(), 
        io_poutputpipeline->psensorinfo->id, io_poutputpipeline->poutputpipeline_ops->outputtype, _pcamera.get(),ucInfo);
    CHK_PTR_AND_RET_S32(upProducer, "CFactory::CreateProducer.");
    HW_NVMEDIA_LOG_DEBUG("Producer is created.\r\n");

    upProducer->SetProfiler(io_poutputpipeline->pprofiler);
    io_poutputpipeline->_peventhandler->vector_pclients.push_back(std::move(upProducer));
    if (NVMEIDA_PIPELIN_VALID == psensorpipelineconfig->enablecuda)
    {
        std::unique_ptr<CConsumer> upCUDAConsumer = CFactory::CreateConsumer(CUDA_CONSUMER, io_poutputpipeline->psensorinfo,
            io_poutputpipeline->poutputpipeline_ops->outputtype, _deviceopenpara.busemailbox, ucInfo, io_poutputpipeline);
        CHK_PTR_AND_RET_S32(upCUDAConsumer, "CFactory::Create CUDA consumer");
        io_poutputpipeline->_peventhandler->vector_pclients.push_back(std::move(upCUDAConsumer));
        HW_NVMEDIA_LOG_DEBUG("CUDA consumer is created.\r\n");
        consumer_num ++;
    }

 if (NVMEIDA_PIPELIN_VALID == psensorpipelineconfig->enablecommon)
    {
        std::unique_ptr<CConsumer> upCommonConsumer = CFactory::CreateConsumer(COMMON_CONSUMER, io_poutputpipeline->psensorinfo,
                                                                               io_poutputpipeline->poutputpipeline_ops->outputtype, _deviceopenpara.busemailbox, ucInfo, io_poutputpipeline);
        CHK_PTR_AND_RET_S32(upCommonConsumer, "CFactory::Create common consumer");
        io_poutputpipeline->_peventhandler->vector_pclients.push_back(std::move(upCommonConsumer));
        HW_NVMEDIA_LOG_DEBUG("Common consumer is created.\r\n");
        consumer_num++;
    }

    if (NVMEIDA_PIPELIN_VALID == psensorpipelineconfig->enablevic)
    {
        std::unique_ptr<CConsumer> upVICConsumer = CFactory::CreateConsumer(VIC_CONSUMER, io_poutputpipeline->psensorinfo,
            io_poutputpipeline->poutputpipeline_ops->outputtype, _deviceopenpara.busemailbox, ucInfo, io_poutputpipeline, encodeType);
        PCHK_PTR_AND_RETURN(upVICConsumer, "CFactory::Create VIC consumer");
        io_poutputpipeline->_peventhandler->vector_pclients.push_back(std::move(upVICConsumer));
        PLOG_DBG("VIC consumer is created.\n");
        consumer_num++;
    }
    else if (NVMEIDA_PIPELIN_VALID == psensorpipelineconfig->enableenc)
    {
        std::unique_ptr<CConsumer> upEncConsumer = CFactory::CreateConsumer(ENC_CONSUMER, io_poutputpipeline->psensorinfo,
                                                                            io_poutputpipeline->poutputpipeline_ops->outputtype, _deviceopenpara.busemailbox, ucInfo, io_poutputpipeline, encodeType);
        CHK_PTR_AND_RET_S32(upEncConsumer, "CFactory::Create encoder consumer");
        io_poutputpipeline->_peventhandler->vector_pclients.push_back(std::move(upEncConsumer));
        HW_NVMEDIA_LOG_DEBUG("Encoder consumer is created.\r\n");
        consumer_num++;
     }
    // printf("consumer_num is %d\n", consumer_num);
     if (consumer_num > 1U)
     {
         CHK_SIPLSTATUS_SENTENCE_AND_RET_S32(CFactory::CreateMulticastBlock(consumer_num, //这个HW_NVMEDIA_NUM_CONSUMERS 得是实际的consumer，不能多
                                                                            io_poutputpipeline->_peventhandler->block_multicast),
                                             "CFactory::CreateMulticastBlock");
         HW_NVMEDIA_LOG_DEBUG("Multicast block is created.\r\n");
     }
    return 0;
    }

hw_ret_s32 HWNvmediaSingleProcessContext::DestroyBlocks(HWNvmediaOutputPipelineContext* io_poutputpipeline)
{
    HW_NVMEDIA_LOG_DEBUG("DestroyBlocks.\r\n");

    if (io_poutputpipeline->_peventhandler->ppoolmanager != nullptr 
        && io_poutputpipeline->_peventhandler->ppoolmanager->GetHandle() != 0U) {
        (void)NvSciStreamBlockDelete(io_poutputpipeline->_peventhandler->ppoolmanager->GetHandle());
    }
    if (io_poutputpipeline->_peventhandler->block_multicast != 0U) {
        (void)NvSciStreamBlockDelete(io_poutputpipeline->_peventhandler->block_multicast);
    }
    if (io_poutputpipeline->_peventhandler->vector_pclients[0] != nullptr 
        && io_poutputpipeline->_peventhandler->vector_pclients[0]->GetHandle() != 0U) {
        (void)NvSciStreamBlockDelete(io_poutputpipeline->_peventhandler->vector_pclients[0]->GetHandle());
    }
    for (uint32_t i = 1U; i < io_poutputpipeline->_peventhandler->vector_pclients.size(); i++) {
        CConsumer* pconsumer = dynamic_cast<CConsumer*>(io_poutputpipeline->_peventhandler->vector_pclients[i].get());
        if (pconsumer != nullptr && pconsumer->GetHandle() != 0U) {
            (void)NvSciStreamBlockDelete(pconsumer->GetHandle());
        }
        if (pconsumer != nullptr && pconsumer->GetQueueHandle() != 0U) {
            (void)NvSciStreamBlockDelete(pconsumer->GetQueueHandle());
        }
    }
    return 0;
}

hw_ret_s32 HWNvmediaSingleProcessContext::ConnectBlocks(HWNvmediaOutputPipelineContext* io_poutputpipeline)
{
    NvSciStreamEventType event;
    NvSciError scierr;

    HW_NVMEDIA_LOG_DEBUG("Connect.\r\n");

    if (HW_NVMEDIA_NUM_CONSUMERS == 1U) {
        CHK_NVSCISTATUS_SENTENCE_AND_RET_S32(NvSciStreamBlockConnect(io_poutputpipeline->_peventhandler->vector_pclients[0]->GetHandle(), 
            io_poutputpipeline->_peventhandler->vector_pclients[1]->GetHandle()),
            ("Producer connect to" + io_poutputpipeline->_peventhandler->vector_pclients[1]->GetName()).c_str());
    }
    else {
        //connect producer with multicast
        CHK_NVSCISTATUS_SENTENCE_AND_RET_S32(NvSciStreamBlockConnect(io_poutputpipeline->_peventhandler->vector_pclients[0]->GetHandle(),
            io_poutputpipeline->_peventhandler->block_multicast),
            "Connect producer to multicast");
        HW_NVMEDIA_LOG_DEBUG("Producer is connected to multicast.\r\n");

        //connect multicast with each consumer
        for (u32 i = 1U; i < io_poutputpipeline->_peventhandler->vector_pclients.size(); i++) {
            CHK_NVSCISTATUS_SENTENCE_AND_RET_S32(NvSciStreamBlockConnect(io_poutputpipeline->_peventhandler->block_multicast, io_poutputpipeline->_peventhandler->vector_pclients[i]->GetHandle()),
                "Multicast connect to consumer");
            HW_NVMEDIA_LOG_DEBUG("Multicast is connected to consumer: %u\r\n", (i - 1));
        }
    }

    HW_NVMEDIA_LOG_UNMASK("Connecting to the stream...\r\n");
    //query producer
    scierr = NvSciStreamBlockEventQuery(io_poutputpipeline->_peventhandler->vector_pclients[0]->GetHandle(),
        QUERY_TIMEOUT_FOREVER, &event);
    CHK_NVSCICONNECT_SENTENCE_AND_RET_S32(scierr, event, "producer");
    HW_NVMEDIA_LOG_DEBUG("Producer is connected.\r\n");

    scierr = NvSciStreamBlockEventQuery(io_poutputpipeline->_peventhandler->ppoolmanager->GetHandle(), QUERY_TIMEOUT_FOREVER, &event);
    CHK_NVSCICONNECT_SENTENCE_AND_RET_S32(scierr, event, "pool");
    HW_NVMEDIA_LOG_DEBUG("Pool is connected.\r\n");

    //query consumers and queues
    for (u32 i = 1U; i < io_poutputpipeline->_peventhandler->vector_pclients.size(); i++) {
        CConsumer* pConsumer = dynamic_cast<CConsumer*>(io_poutputpipeline->_peventhandler->vector_pclients[i].get());
        scierr = NvSciStreamBlockEventQuery(pConsumer->GetQueueHandle(), QUERY_TIMEOUT_FOREVER, &event);
        CHK_NVSCICONNECT_SENTENCE_AND_RET_S32(scierr, event, "queue");
        HW_NVMEDIA_LOG_DEBUG("Queue:%u is connected.\r\n", (i - 1));

        scierr = NvSciStreamBlockEventQuery(pConsumer->GetHandle(), QUERY_TIMEOUT_FOREVER, &event);
        CHK_NVSCICONNECT_SENTENCE_AND_RET_S32(scierr, event, "consumer");
        HW_NVMEDIA_LOG_DEBUG("Consumer:%u is connected.\r\n", (i - 1));
    }

    //query multicast
    if (io_poutputpipeline->_peventhandler->block_multicast != 0U) {
        scierr = NvSciStreamBlockEventQuery(io_poutputpipeline->_peventhandler->block_multicast, QUERY_TIMEOUT_FOREVER, &event);
        CHK_NVSCICONNECT_SENTENCE_AND_RET_S32(scierr, event, "multicast");
        HW_NVMEDIA_LOG_DEBUG("Multicast is connected.\r\n");
    }

    HW_NVMEDIA_LOG_UNMASK("All blocks are connected to the stream!\r\n");
    return 0;
}

hw_ret_s32 HWNvmediaSingleProcessContext::InitBlocks(HWNvmediaOutputPipelineContext* io_poutputpipeline)
{
    HW_NVMEDIA_LOG_DEBUG("InitBlocks.\r\n");

    CHK_SIPLSTATUS_SENTENCE_AND_RET_S32(io_poutputpipeline->_peventhandler->ppoolmanager->Init(),
        "Pool Init");

    for (auto& upClient : io_poutputpipeline->_peventhandler->vector_pclients) {
        CHK_SIPLSTATUS_SENTENCE_AND_RET_S32(upClient->Init(_scibufmodule, _scisyncmodule),
            (upClient->GetName() + " Init").c_str());
    }
    return 0;
}

hw_ret_s32 HWNvmediaSingleProcessContext::GetEventHandlerVector_Reconcile(HWNvmediaOutputPipelineContext* io_poutputpipeline,
    std::vector<CEventHandler*>& i_vector_eventhandlers)
{
    i_vector_eventhandlers.push_back(io_poutputpipeline->_peventhandler->ppoolmanager.get());
    for (auto& upclients : io_poutputpipeline->_peventhandler->vector_pclients) {
        i_vector_eventhandlers.push_back(upclients.get());
    }
    return 0;
}

hw_ret_s32 HWNvmediaSingleProcessContext::GetEventHandlerVector_Start(HWNvmediaOutputPipelineContext* io_poutputpipeline,
    std::vector<CEventHandler*>& i_vector_eventhandlers)
{
    for (auto& upclients : io_poutputpipeline->_peventhandler->vector_pclients) {
        i_vector_eventhandlers.push_back(upclients.get());
    }
    return 0;
}
