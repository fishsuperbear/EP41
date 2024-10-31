#include "hw_nvmedia_common_impl.h"
#include "hw_nvmedia_eventhandler_impl.h"
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include "halnode.h"

hw_ret_s32 HWNvmediaIpcConsumerContext::Pipeline_Open(struct hw_video_blockspipelineconfig_t *i_pblockspipelineconfig,
                                                      struct hw_video_blockspipeline_ops_t **o_ppblockspipeline_ops) {
    HW_NVMEDIA_LOG_UNMASK("HWNvmediaIpcConsumerContext::Pipeline_Open Enter!\r\n");

    CHK_LOG_SENTENCE_HW_RET_S32(pipelineopen(i_pblockspipelineconfig));
    CHK_LOG_SENTENCE_HW_RET_S32(pipelineprepare());
    *o_ppblockspipeline_ops = _pblockspipeline_ops;
    return 0;
}

hw_ret_s32 HWNvmediaIpcConsumerContext::Pipeline_Close() {
    HW_NVMEDIA_LOG_UNMASK("HWNvmediaIpcConsumerContext::Pipeline_Close Enter!\r\n");

    CHK_LOG_SENTENCE_HW_RET_S32(pipelineunprepare());
    CHK_LOG_SENTENCE_HW_RET_S32(pipelineclose());
    return 0;
}

hw_ret_s32 HWNvmediaIpcConsumerContext::Pipeline_Start() {
    HW_NVMEDIA_LOG_UNMASK("HWNvmediaIpcConsumerContext::Pipeline_Start Enter!\r\n");

    CHK_LOG_SENTENCE_HW_RET_S32(pipelinestart());
    return 0;
}

hw_ret_s32 HWNvmediaIpcConsumerContext::Pipeline_Stop() {
    HW_NVMEDIA_LOG_UNMASK("HWNvmediaIpcConsumerContext::Pipeline_Stop Enter!\r\n");

    CHK_LOG_SENTENCE_HW_RET_S32(pipelinestop());
    return 0;
}

hw_ret_s32 HWNvmediaIpcConsumerContext::CreateBlocks(HWNvmediaOutputPipelineContext *io_poutputpipeline) {
    HW_NVMEDIA_LOG_UNMASK("HWNvmediaIpcConsumerContext::CreateBlocks Enter!\r\n");

    int encodeType = HW_VIDEO_REGDATACB_TYPE_HEVC;
    UseCaseInfo ucInfo;

    hw_video_sensorpipelineconfig_t *psensorpipelineconfig = &_blockspipelineconfig.parrayblock[io_poutputpipeline->poutputpipeline_ops->blockindex].parraysensor[io_poutputpipeline->poutputpipeline_ops->sensorindex];
    for (u32 i = 0; i < psensorpipelineconfig->datacbsconfig.arraynumdatacbs; i++) {
        if (psensorpipelineconfig->datacbsconfig.parraydatacbs[i].type == HW_VIDEO_REGDATACB_TYPE_HEVC) {
            encodeType = HW_VIDEO_REGDATACB_TYPE_HEVC;
        } else if (psensorpipelineconfig->datacbsconfig.parraydatacbs[i].type == HW_VIDEO_REGDATACB_TYPE_AVC) {
            encodeType = HW_VIDEO_REGDATACB_TYPE_AVC;
        }
    }
    if (psensorpipelineconfig->bcaptureoutputrequested) {
        // enable ICP
        ucInfo.isMultiElems = false;
        ucInfo.isEnableICP = true;
    } else if (psensorpipelineconfig->bisp0outputrequested && psensorpipelineconfig->bisp1outputrequested) {
        // enable ISP0 and ISP1
        ucInfo.isMultiElems = true;
    } else if (psensorpipelineconfig->bisp0outputrequested && !psensorpipelineconfig->bisp1outputrequested) {
        // enable ISP0
        ucInfo.isMultiElems = false;
    } else if (!psensorpipelineconfig->bisp0outputrequested && psensorpipelineconfig->bisp1outputrequested) {
        // enable ISP1
        ucInfo.isMultiElems = false;
    }

    ConsumerType apptype = CUDA_CONSUMER;
    struct consumer_info o_consumer_info = {
       .flag = 1,
    };

    if (_deviceopenpara.apptype == HW_NVMEDIA_APPTYPE_IPC_CONSUMER_CUDA) {
	o_consumer_info.data_info.bgpudata = 1;
        consumer_start(io_poutputpipeline->psensorinfo->id);//lattach attach
        {//get idx
            io_poutputpipeline->_client_fd = socket(AF_UNIX, SOCK_STREAM, 0);
            if (io_poutputpipeline->_client_fd == -1) {
                return -1;
            }
            char socket_path[100];
            sprintf(socket_path, "/tmp/.cam_hal_reattach_%d",io_poutputpipeline->psensorinfo->id);
            struct sockaddr_un server_addr;
            memset(&server_addr, 0, sizeof(server_addr));
            server_addr.sun_family = AF_UNIX;
            strncpy(server_addr.sun_path, socket_path, sizeof(server_addr.sun_path) - 1);
            if (connect(io_poutputpipeline->_client_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
                close(io_poutputpipeline->_client_fd);
                return -1;
            }
            const char* command = "getidx";
            if (write(io_poutputpipeline->_client_fd, command, strlen(command)) == -1) {
                close(io_poutputpipeline->_client_fd);
                return -1;
            }
            char buffer[8];
            int read_len = read(io_poutputpipeline->_client_fd, buffer, sizeof(buffer) - 1);
            if (read_len == -1) {
                close(io_poutputpipeline->_client_fd);
                return -1;
            }
            buffer[read_len] = '\0';
            _cuda_idx = atoi(buffer);
            if(_cuda_idx<1){
                HW_NVMEDIA_LOG_ERR("getidx from producer failed.No enough resource\n");
                close(io_poutputpipeline->_client_fd);
                return -1;
            }
        }
        apptype = CUDA_CONSUMER;
        auto m_upConsumer = CFactory::CreateConsumer(apptype,
                                                     io_poutputpipeline->psensorinfo,
                                                     io_poutputpipeline->poutputpipeline_ops->outputtype,
                                                     _deviceopenpara.busemailbox,
                                                     ucInfo,
                                                     io_poutputpipeline);
        HW_NVMEDIA_LOG_DEBUG("HWNvmediaIpcConsumerContext::CreateBlocks CFactory::CreateConsumer=%d\n", apptype);
        io_poutputpipeline->_peventhandler->vector_pclients.push_back(std::move(m_upConsumer));
    } else if (_deviceopenpara.apptype == HW_NVMEDIA_APPTYPE_IPC_CONSUMER_ENC) {
        HW_NVMEDIA_LOG_INFO("Encoder consumer is created.\r\n");
	o_consumer_info.data_info.bgpudata = 0;
        apptype = ENC_CONSUMER;
        if(psensorpipelineconfig->bcaptureoutputrequested){
            auto upVicConsumer = CFactory::CreateConsumer(VIC_CONSUMER, io_poutputpipeline->psensorinfo,
                    io_poutputpipeline->poutputpipeline_ops->outputtype,
                    _deviceopenpara.busemailbox,
                    ucInfo, io_poutputpipeline, encodeType);
            CHK_PTR_AND_RET_S32(upVicConsumer, "CFactory::Create VIC consumer");
            io_poutputpipeline->_peventhandler->vector_pclients.push_back(std::move(upVicConsumer));
            HW_NVMEDIA_LOG_DEBUG("VIC consumer is created.\r\n");
        }else{

            auto upEncConsumer = CFactory::CreateConsumer(ENC_CONSUMER, io_poutputpipeline->psensorinfo,
                    io_poutputpipeline->poutputpipeline_ops->outputtype,
                    _deviceopenpara.busemailbox,
                    ucInfo, io_poutputpipeline, encodeType);
            CHK_PTR_AND_RET_S32(upEncConsumer, "CFactory::Create encoder consumer");
            io_poutputpipeline->_peventhandler->vector_pclients.push_back(std::move(upEncConsumer));
            HW_NVMEDIA_LOG_DEBUG("Encoder consumer is created.\r\n");
        }
        /* auto upEncConsumer = CFactory::CreateConsumer(ENC_CONSUMER, io_poutputpipeline->psensorinfo, */
        /*                                               io_poutputpipeline->poutputpipeline_ops->outputtype, */
        /*                                               _deviceopenpara.busemailbox, */
        /*                                               ucInfo, io_poutputpipeline, encodeType); */
        /* CHK_PTR_AND_RET_S32(upEncConsumer, "CFactory::Create encoder consumer"); */
        /* io_poutputpipeline->_peventhandler->vector_pclients.push_back(std::move(upEncConsumer)); */
        /* HW_NVMEDIA_LOG_DEBUG("Encoder consumer is created.\r\n"); */
    }
    // TODO(zax): channel name define need optimize. name set should wrapper in API, do not set by user.
    /*
    * We must use the physical sensor id (psensorinfo->id) as the input parameter of CreateIPCProducer constructor.
    */
    std::string dstChannel = "nvscistream_" + std::to_string(io_poutputpipeline->psensorinfo->id * NUM_IPC_CONSUMERS * 2 + 2 * (apptype==ENC_CONSUMER?0:_cuda_idx) + 1);
    HW_NVMEDIA_LOG_DEBUG("dstChannel name: %s\r\n", dstChannel.c_str());
    auto status = CFactory::CreateIpcBlock(_scisyncmodule, _scibufmodule, dstChannel.c_str(), false, &io_poutputpipeline->_peventhandler->pdstIpcHandle);
    if (status != SIPLStatus::NVSIPL_STATUS_OK) {
        HW_NVMEDIA_LOG_ERR("HWNvmediaIpcConsumerContext::CreateBlocks CFactory::CreateIpcBlock failed!\r\n");
        return -1;
    }
    set_consumer_info(io_poutputpipeline->psensorinfo->id,(apptype==ENC_CONSUMER?0:_cuda_idx),&o_consumer_info);
    HW_NVMEDIA_LOG_INFO("HWNvmediaIpcConsumerContext::CreateBlocks CFactory::CreateIpcBlock success\r\n");

    return 0;
}

hw_ret_s32 HWNvmediaIpcConsumerContext::DestroyBlocks(HWNvmediaOutputPipelineContext *io_poutputpipeline) {
    HW_NVMEDIA_LOG_UNMASK("HWNvmediaIpcConsumerContext::DestroyBlocks Enter!\r\n");

    if (io_poutputpipeline->_peventhandler->ppoolmanager != nullptr && io_poutputpipeline->_peventhandler->ppoolmanager->GetHandle() != 0U) {
        (void)NvSciStreamBlockDelete(io_poutputpipeline->_peventhandler->ppoolmanager->GetHandle());
    }
    if (io_poutputpipeline->_peventhandler->block_multicast != 0U) {
        (void)NvSciStreamBlockDelete(io_poutputpipeline->_peventhandler->block_multicast);
    }
    if (io_poutputpipeline->_peventhandler->vector_pclients[0] != nullptr && io_poutputpipeline->_peventhandler->vector_pclients[0]->GetHandle() != 0U) {
        (void)NvSciStreamBlockDelete(io_poutputpipeline->_peventhandler->vector_pclients[0]->GetHandle());
    }

    for (uint32_t i = 1U; i < io_poutputpipeline->_peventhandler->vector_pclients.size(); i++) {
        CConsumer *pconsumer = dynamic_cast<CConsumer *>(io_poutputpipeline->_peventhandler->vector_pclients[i].get());
        if (pconsumer != nullptr && pconsumer->GetHandle() != 0U) {
            (void)NvSciStreamBlockDelete(pconsumer->GetHandle());
        }
        if (pconsumer != nullptr && pconsumer->GetQueueHandle() != 0U) {
            (void)NvSciStreamBlockDelete(pconsumer->GetQueueHandle());
        }
    }

    if (io_poutputpipeline->_peventhandler->pdstIpcHandle != 0U) {
        (void)NvSciStreamBlockDelete(io_poutputpipeline->_peventhandler->pdstIpcHandle);
    }

    if(io_poutputpipeline->_client_fd > 0){
        close(io_poutputpipeline->_client_fd);
    }

    return 0;
}

hw_ret_s32 HWNvmediaIpcConsumerContext::ConnectBlocks(HWNvmediaOutputPipelineContext *io_poutputpipeline) {
    HW_NVMEDIA_LOG_UNMASK("HWNvmediaIpcConsumerContext::ConnectBlocks Enter!\r\n");

    NvSciStreamEventType event;
    CConsumer *pconsumer = dynamic_cast<CConsumer *>(io_poutputpipeline->_peventhandler->vector_pclients[0].get());

    auto sciErr = NvSciStreamBlockConnect(io_poutputpipeline->_peventhandler->pdstIpcHandle, pconsumer->GetHandle());
    HW_NVMEDIA_LOG_INFO("HWNvmediaIpcConsumerContext::ConnectBlocks NvSciStreamBlockConnect\n");
    CHK_NVSCISTATUS_SENTENCE_AND_RET_S32(sciErr, "Connect blocks: dstIpc - consumer");
    HW_NVMEDIA_LOG_INFO("%s is connecting to the stream...\r\n", pconsumer->GetName().c_str());
    if (_deviceopenpara.apptype == HW_NVMEDIA_APPTYPE_IPC_CONSUMER_CUDA) {
        //attach to producer
        char command[128];
        sprintf(command,"attach:%d",_cuda_idx);
        if (write(io_poutputpipeline->_client_fd, command, strlen(command)) == -1) {
            /* cerr << "Failed to send command to server" << endl; */
            HW_NVMEDIA_LOG_ERR("attach to producer failed\r\n");
            close(io_poutputpipeline->_client_fd);
            return -1;
        }
    }

    HW_NVMEDIA_LOG_INFO("Query ipc dst connection\r\n");
    sciErr = NvSciStreamBlockEventQuery(io_poutputpipeline->_peventhandler->pdstIpcHandle, QUERY_TIMEOUT_FOREVER, &event);
    HW_NVMEDIA_LOG_INFO("HWNvmediaIpcConsumerContext::ConnectBlocks NvSciStreamBlockEventQuery\n");
    CheckNvSciConnectAndReturn(sciErr, event, "ipc dst", pconsumer->GetName().c_str());
    HW_NVMEDIA_LOG_INFO("Ipc dst is connected\r\n");

    // query consumer and queue
    HW_NVMEDIA_LOG_INFO("Query queue connection\r\n");
    sciErr = NvSciStreamBlockEventQuery(pconsumer->GetQueueHandle(), QUERY_TIMEOUT_FOREVER, &event);
    HW_NVMEDIA_LOG_INFO("HWNvmediaIpcConsumerContext::ConnectBlocks NvSciStreamBlockEventQuery\n");
    CheckNvSciConnectAndReturn(sciErr, event, "queue", pconsumer->GetName().c_str());
    HW_NVMEDIA_LOG_INFO("Queue is connected\r\n");

    HW_NVMEDIA_LOG_INFO("Query consumer connection\r\n");
    sciErr = NvSciStreamBlockEventQuery(pconsumer->GetHandle(), QUERY_TIMEOUT_FOREVER, &event);
    HW_NVMEDIA_LOG_INFO("HWNvmediaIpcConsumerContext::ConnectBlocks NvSciStreamBlockEventQuery\n");
    CheckNvSciConnectAndReturn(sciErr, event, "consumer", pconsumer->GetName().c_str());
    HW_NVMEDIA_LOG_INFO("Consumer is connected\r\n");

    HW_NVMEDIA_LOG_INFO("%s is connected to the stream\r\n", pconsumer->GetName().c_str());
    return 0;
}

hw_ret_s32 HWNvmediaIpcConsumerContext::InitBlocks(HWNvmediaOutputPipelineContext *io_poutputpipeline) {
    HW_NVMEDIA_LOG_UNMASK("HWNvmediaIpcConsumerContext::InitBlocks Enter!\r\n");

    CConsumer *pconsumer = dynamic_cast<CConsumer *>(io_poutputpipeline->_peventhandler->vector_pclients[0].get());
    auto status = pconsumer->Init(_scibufmodule, _scisyncmodule);
    if (status != SIPLStatus::NVSIPL_STATUS_OK) {
        HW_NVMEDIA_LOG_ERR("Consumer Init failed!\r\n");
        return -1;
    }
    HW_NVMEDIA_LOG_INFO("Consumer Init success\r\n");

    return 0;
}

hw_ret_s32 HWNvmediaIpcConsumerContext::GetEventHandlerVector_Reconcile(HWNvmediaOutputPipelineContext *io_poutputpipeline,
                                                                        std::vector<CEventHandler *> &i_vector_eventhandlers) {
    HW_NVMEDIA_LOG_UNMASK("HWNvmediaIpcConsumerContext::GetEventHandlerVector_Reconcile Enter!\r\n");

    CConsumer *pconsumer = dynamic_cast<CConsumer *>(io_poutputpipeline->_peventhandler->vector_pclients[0].get());
    if (pconsumer == nullptr) {
        HW_NVMEDIA_LOG_ERR("pconsumer == nullptr\r\n");
        return -1;
    }
    i_vector_eventhandlers.push_back(pconsumer);
    return 0;
}

hw_ret_s32 HWNvmediaIpcConsumerContext::GetEventHandlerVector_Start(HWNvmediaOutputPipelineContext *io_poutputpipeline,
                                                                    std::vector<CEventHandler *> &i_vector_eventhandlers) {
    HW_NVMEDIA_LOG_UNMASK("HWNvmediaIpcConsumerContext::GetEventHandlerVector_Start Enter!\r\n");

    CConsumer *pconsumer = dynamic_cast<CConsumer *>(io_poutputpipeline->_peventhandler->vector_pclients[0].get());
    i_vector_eventhandlers.push_back(pconsumer);
    return 0;
}

hw_ret_s32 HWNvmediaIpcConsumerContext::CheckNvSciConnectAndReturn(NvSciError sciErr, NvSciStreamEventType event, std::string api, std::string name) {
    if (NvSciError_Success != sciErr) {
        HW_NVMEDIA_LOG_ERR("%s:  connect failed! %d\r\n", name.c_str(), sciErr);
        return -1;
    }
    if (event != NvSciStreamEventType_Connected) {
        HW_NVMEDIA_LOG_UNMASK("%s: %s didn't receive connected event!\r\n", name.c_str(), api.c_str());
        return -1;
    }

    return 0;
}
