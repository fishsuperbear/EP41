#include "hw_nvmedia_common_impl.h"
#include "hw_nvmedia_eventhandler_impl.h"
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <fcntl.h>
#include <thread>
#include "halnode.h"

std::string m_srcChannels[NUM_IPC_CONSUMERS];

hw_ret_s32 HWNvmediaIpcProducerContext::Pipeline_Open(struct hw_video_blockspipelineconfig_t *i_pblockspipelineconfig,
                                                      struct hw_video_blockspipeline_ops_t **o_ppblockspipeline_ops) {
    HW_NVMEDIA_LOG_UNMASK("HWNvmediaIpcProducerContext::Pipeline_Open Enter!\r\n");

    CHK_LOG_SENTENCE_HW_RET_S32(pipelineopen(i_pblockspipelineconfig));
    CHK_LOG_SENTENCE_HW_RET_S32(pipelineprepare());
    *o_ppblockspipeline_ops = _pblockspipeline_ops;
    return 0;
}

hw_ret_s32 HWNvmediaIpcProducerContext::Pipeline_Close() {
    HW_NVMEDIA_LOG_UNMASK("HWNvmediaIpcProducerContext::Pipeline_Close Enter!\r\n");

    CHK_LOG_SENTENCE_HW_RET_S32(pipelineunprepare());
    CHK_LOG_SENTENCE_HW_RET_S32(pipelineclose());
    return 0;
}

hw_ret_s32 HWNvmediaIpcProducerContext::Pipeline_Start() {
    HW_NVMEDIA_LOG_UNMASK("HWNvmediaIpcProducerContext::Pipeline_Start Enter!\r\n");

    CHK_LOG_SENTENCE_HW_RET_S32(pipelinestart());
    return 0;
}

hw_ret_s32 HWNvmediaIpcProducerContext::Pipeline_Stop() {
    HW_NVMEDIA_LOG_UNMASK("HWNvmediaIpcProducerContext::Pipeline_Stop Enter!\r\n");

    CHK_LOG_SENTENCE_HW_RET_S32(pipelinestop());
    return 0;
}

hw_ret_s32 HWNvmediaIpcProducerContext::CreateBlocks(HWNvmediaOutputPipelineContext *io_poutputpipeline) {
    HW_NVMEDIA_LOG_UNMASK("HWNvmediaIpcProducerContext::CreateBlocks Enter!\r\n");

    bool o_enableLateAttach = io_poutputpipeline->GetEnableLateAttach();
    uint32_t o_uLateConsCount = io_poutputpipeline->GetLateConsCount();
    // int encodeType = HW_VIDEO_REGDATACB_TYPE_AVC;
    UseCaseInfo ucInfo;

    std::shared_ptr<CAttributeProvider> attrProvider = nullptr;
    if (o_enableLateAttach)
    {
        attrProvider = CFactory::CreateAttributeProvider(_scisyncmodule, _scibufmodule);
        PCHK_PTR_AND_RETURN(attrProvider, "CFactory::CreateAttributeProvider");
    }

    hw_video_sensorpipelineconfig_t *psensorpipelineconfig = &_blockspipelineconfig.parrayblock[io_poutputpipeline->poutputpipeline_ops->blockindex].parraysensor[io_poutputpipeline->poutputpipeline_ops->sensorindex];
    // for (u32 i = 0; i < psensorpipelineconfig->datacbsconfig.arraynumdatacbs; i++)
    // {
    //     if (psensorpipelineconfig->datacbsconfig.parraydatacbs[i].type == HW_VIDEO_REGDATACB_TYPE_HEVC)
    //     {
    //         encodeType = HW_VIDEO_REGDATACB_TYPE_HEVC;
    //     }
    //     else if (psensorpipelineconfig->datacbsconfig.parraydatacbs[i].type == HW_VIDEO_REGDATACB_TYPE_AVC)
    //     {
    //         encodeType = HW_VIDEO_REGDATACB_TYPE_AVC;
    //     }
    // }
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

    io_poutputpipeline->_peventhandler->ppoolmanager = CFactory::CreatePoolManager(io_poutputpipeline->psensorinfo->id,
                                                                                   MAX_NUM_PACKETS,attrProvider);
    PLOG_DBG("HWNvmediaIpcProducerContext::CreateBlocks CFactory::CreatePoolManager\n");
    /*
    * We must use the physical sensor id (psensorinfo->id) as the input parameter of CreateIPCProducer constructor.
    */
    std::unique_ptr<CProducer> m_upProducer = CFactory::CreateIPCProducer(io_poutputpipeline->_peventhandler->ppoolmanager->GetHandle(),
                                                                          io_poutputpipeline->psensorinfo->id,
                                                                          io_poutputpipeline->poutputpipeline_ops->outputtype,
                                                                          _pcamera.get(),
                                                                          ucInfo,attrProvider);
    PLOG_DBG("HWNvmediaIpcProducerContext::CreateBlocks CFactory::CreateProducer\n");
    m_upProducer->SetProfiler(io_poutputpipeline->pprofiler);
    io_poutputpipeline->_peventhandler->vector_pclients.push_back(std::move(m_upProducer));

    if (NUM_IPC_CONSUMERS > 1U) {
        auto status = CFactory::CreateMulticastBlock(NUM_IPC_CONSUMERS, io_poutputpipeline->_peventhandler->block_multicast);
        if (status != SIPLStatus::NVSIPL_STATUS_OK) {
            PLOG_DBG("HWNvmediaIpcConsumerContext::CreateBlocks CFactory::CreateMulticastBlock failed!\r\n");
            return -1;
        }
        PLOG_DBG("HWNvmediaIpcConsumerContext::CreateBlocks CFactory::CreateMulticastBlock success\r\n");
    }

    for (auto i = 0U; i < NUM_IPC_CONSUMERS - o_uLateConsCount; i++) {
        std::string srcChannel = "nvscistream_" + std::to_string(io_poutputpipeline->psensorinfo->id * NUM_IPC_CONSUMERS * 2 + 2 * i + 0);
        PLOG_DBG("HWNvmediaIpcConsumerContext::CreateBlocks srcChannel[%d]: %s\n", i, srcChannel.c_str());
        auto status = CFactory::CreateIpcBlock(_scisyncmodule, _scibufmodule, srcChannel.c_str(), true, &io_poutputpipeline->_peventhandler->psrcIpcHandles[i]);
        if (status != SIPLStatus::NVSIPL_STATUS_OK) {
            PLOG_DBG("HWNvmediaIpcConsumerContext::CreateBlocks CFactory::CreateIpcBlock failed!\r\n");
            return -1;
        }
        PLOG_DBG("HWNvmediaIpcConsumerContext::CreateBlocks CFactory::CreateIpcBlock success\r\n");
    }
    {
        io_poutputpipeline->_server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (io_poutputpipeline->_server_fd == -1) {
            cerr << "Failed to create socket" << endl;
            return -1;
        }

        // bind socket
        char socket_path[100];
        sprintf(socket_path, "/tmp/.cam_hal_reattach_%d",io_poutputpipeline->psensorinfo->id);
        //char cmd[200];
        //sprintf(cmd,"rm -rf %s",socket_path);

        unlink(socket_path);
        //int cmdret = system(cmd);
        //HW_NVMEDIA_LOG_UNMASK("system cmd=%d\r\n", cmdret);

        struct sockaddr_un server_addr;
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sun_family = AF_UNIX;
        strncpy(server_addr.sun_path, socket_path, sizeof(server_addr.sun_path) - 1);
        if (bind(io_poutputpipeline->_server_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
            cerr << "Failed to bind socket" << endl;
            close(io_poutputpipeline->_server_fd);
            unlink(socket_path);
            return -1;
        }
    }

    return 0;
}

hw_ret_s32 HWNvmediaIpcProducerContext::DestroyBlocks(HWNvmediaOutputPipelineContext *io_poutputpipeline) {
    PLOG_DBG("HWNvmediaIpcProducerContext::DestroyBlocks Enter!\r\n");

    if (io_poutputpipeline->_peventhandler->ppoolmanager != nullptr && io_poutputpipeline->_peventhandler->ppoolmanager->GetHandle() != 0U) {
        (void)NvSciStreamBlockDelete(io_poutputpipeline->_peventhandler->ppoolmanager->GetHandle());
    }
    if (io_poutputpipeline->_peventhandler->block_multicast != 0U) {
        (void)NvSciStreamBlockDelete(io_poutputpipeline->_peventhandler->block_multicast);
    }
    if (io_poutputpipeline->_peventhandler->vector_pclients[0] != nullptr && io_poutputpipeline->_peventhandler->vector_pclients[0]->GetHandle() != 0U) {
        (void)NvSciStreamBlockDelete(io_poutputpipeline->_peventhandler->vector_pclients[0]->GetHandle());
    }

    for (auto i = 0U; i < NUM_IPC_CONSUMERS; i++) {
        if (io_poutputpipeline->_peventhandler->psrcIpcHandles[i] != 0U) {
            (void)NvSciStreamBlockDelete(io_poutputpipeline->_peventhandler->psrcIpcHandles[i]);
        }
    }

    return 0;
}

hw_ret_s32 HWNvmediaIpcProducerContext::ConnectBlocks(HWNvmediaOutputPipelineContext *io_poutputpipeline) {
    PLOG_DBG("HWNvmediaIpcProducerContext::ConnectBlocks Enter!\r\n");

    NvSciStreamEventType event;
    CProducer *pProducer = dynamic_cast<CProducer *>(io_poutputpipeline->_peventhandler->vector_pclients[0].get());
    if (pProducer == nullptr) {
        PLOG_DBG("HWNvmediaIpcProducerContext::ConnectBlocks dynamic_cast to CProducer* failed!\n");
        return -1;
    }
    //producer ready for connect
    if(m_pevent != nullptr)
    {
        hw_plat_event_set(m_pevent);
    }

    if (NUM_IPC_CONSUMERS == 1U) {
        auto sciErr = NvSciStreamBlockConnect(pProducer->GetHandle(), io_poutputpipeline->_peventhandler->psrcIpcHandles[0]);
        CHK_NVSCISTATUS_SENTENCE_AND_RET_S32(sciErr, "Producer connect to ipc src");
        PLOG_DBG("Producer is connected to ipc src.\n");
    } else {
        // connect producer with multicast
        auto sciErr = NvSciStreamBlockConnect(pProducer->GetHandle(), io_poutputpipeline->_peventhandler->block_multicast);
        CHK_NVSCISTATUS_SENTENCE_AND_RET_S32(sciErr, "Connect producer to multicast");
        PLOG_DBG("Producer is connected to multicast.\n");

        for (auto i = 0U; i < NUM_IPC_CONSUMERS - io_poutputpipeline->GetLateConsCount(); i++) {
            PLOG_DBG("Block[%u]Sensor[%u], Multicast try to connect to ipc src: %u\n", io_poutputpipeline->poutputpipeline_ops->blockindex, io_poutputpipeline->poutputpipeline_ops->sensorindex, i);
            sciErr = NvSciStreamBlockConnect(io_poutputpipeline->_peventhandler->block_multicast, io_poutputpipeline->_peventhandler->psrcIpcHandles[i]);
            CHK_NVSCISTATUS_SENTENCE_AND_RET_S32(sciErr, "Multicast connect to ipc src");
            PLOG_DBG("Multicast is connected to ipc src: %u\n", i);
        }
    }

    NvSciError sciErr = NvSciError_Success;
    //indicate Multicast to proceed with the initialization and streaming with the connected consumers
    if (io_poutputpipeline->_peventhandler->block_multicast != 0U && io_poutputpipeline->GetLateConsCount() > 0)
    {
        sciErr = NvSciStreamBlockSetupStatusSet(io_poutputpipeline->_peventhandler->block_multicast, NvSciStreamSetup_Connect, true);
        CHK_NVSCISTATUS_SENTENCE_AND_RET_S32(sciErr, "Multicast status set to NvSciStreamSetup_Connect");
    }


    PLOG_DBG("Producer is connecting to the stream...\n");
    // query producer
    PLOG_DBG("Query producer connection.\n");
    PLOG_DBG("Query producer connection.\n");
    sciErr = NvSciStreamBlockEventQuery(pProducer->GetHandle(), QUERY_TIMEOUT_FOREVER, &event);
    CheckNvSciConnectAndReturn(sciErr, event, "Producer", pProducer->GetName().c_str());
    PLOG_DBG("Producer is connected.\n");
    PLOG_DBG("Producer is connected.\n");

    PLOG_DBG("Query pool connection.\n");
    PLOG_DBG("Query pool connection.\n");
    sciErr = NvSciStreamBlockEventQuery(io_poutputpipeline->_peventhandler->ppoolmanager->GetHandle(), QUERY_TIMEOUT_FOREVER, &event);
    CheckNvSciConnectAndReturn(sciErr, event, "Pool", pProducer->GetName().c_str());
    PLOG_DBG("Pool is connected.\n");
    PLOG_DBG("Pool is connected.\n");

    PLOG_DBG("Query ipc src connection.\n");
    PLOG_DBG("Query ipc src connection.\n");
    // query consumers and queues
    for (auto i = 0U; i < NUM_IPC_CONSUMERS - io_poutputpipeline->GetLateConsCount(); i++) {
        sciErr = NvSciStreamBlockEventQuery(io_poutputpipeline->_peventhandler->psrcIpcHandles[i], QUERY_TIMEOUT_FOREVER, &event);
        CheckNvSciConnectAndReturn(sciErr, event, "Ipc src", pProducer->GetName().c_str());
        PLOG_DBG("Ipc src: %u is connected.\n", i);
        PLOG_DBG("Ipc src: %u is connected.\n", i);
    }

    // query multicast
    if (io_poutputpipeline->_peventhandler->block_multicast != 0U) {
        PLOG_DBG("Query multicast block.\n");
        PLOG_DBG("Query multicast block.\n");
        sciErr = NvSciStreamBlockEventQuery(io_poutputpipeline->_peventhandler->block_multicast, QUERY_TIMEOUT_FOREVER, &event);
        CheckNvSciConnectAndReturn(sciErr, event, "Multicast", pProducer->GetName().c_str());
        PLOG_DBG("Multicast block is connected.\n");
        PLOG_DBG("Multicast block is connected.\n");
    }

    std::thread attachEventThread(AttachEventListen,this, io_poutputpipeline);
    attachEventThread.detach();
    PLOG_DBG("Producer is connected to the stream!\n");
    return 0;
}

hw_ret_s32 HWNvmediaIpcProducerContext::InitBlocks(HWNvmediaOutputPipelineContext *io_poutputpipeline) {
    PLOG_DBG("HWNvmediaIpcProducerContext::InitBlocks Enter!\r\n");

    auto status = io_poutputpipeline->_peventhandler->ppoolmanager->Init(io_poutputpipeline->GetEnableLateAttach());
    if (status != SIPLStatus::NVSIPL_STATUS_OK) {
        PLOG_DBG("PoolManager Init failed!\r\n");
        return -1;
    }
    PLOG_DBG("PoolManager Init success\r\n");

    CProducer *pProducer = dynamic_cast<CProducer *>(io_poutputpipeline->_peventhandler->vector_pclients[0].get());
    status = pProducer->Init(_scibufmodule, _scisyncmodule,io_poutputpipeline->GetEnableLateAttach());
    if (status != SIPLStatus::NVSIPL_STATUS_OK) {
        PLOG_DBG("Producer Init failed!\r\n");
        return -1;
    }
    PLOG_DBG("Producer Init success\r\n");

    return 0;
}

hw_ret_s32 HWNvmediaIpcProducerContext::GetEventHandlerVector_Reconcile(HWNvmediaOutputPipelineContext *io_poutputpipeline,
                                                                        std::vector<CEventHandler *> &i_vector_eventhandlers) {
    PLOG_DBG("HWNvmediaIpcProducerContext::GetEventHandlerVector_Reconcile Enter!\r\n");

    i_vector_eventhandlers.push_back(io_poutputpipeline->_peventhandler->ppoolmanager.get());

    CProducer *pProducer = dynamic_cast<CProducer *>(io_poutputpipeline->_peventhandler->vector_pclients[0].get());
    i_vector_eventhandlers.push_back(pProducer);

    return 0;
}

hw_ret_s32 HWNvmediaIpcProducerContext::GetEventHandlerVector_Start(HWNvmediaOutputPipelineContext *io_poutputpipeline,
                                                                    std::vector<CEventHandler *> &i_vector_eventhandlers) {
    PLOG_DBG("HWNvmediaIpcProducerContext::GetEventHandlerVector_Start Enter!\r\n");

    CProducer *pProducer = dynamic_cast<CProducer *>(io_poutputpipeline->_peventhandler->vector_pclients[0].get());
    i_vector_eventhandlers.push_back(pProducer);

    return 0;
}

hw_ret_s32 HWNvmediaIpcProducerContext::CheckNvSciConnectAndReturn(NvSciError sciErr, NvSciStreamEventType event, std::string api, std::string name) {
    if (NvSciError_Success != sciErr) {
        PLOG_DBG("%s:  connect failed! %d\r\n", name.c_str(), sciErr);
        return -1;
    }
    if (event != NvSciStreamEventType_Connected) {
        PLOG_DBG("%s: %s didn't receive connected event!\r\n", name.c_str(), api.c_str());
        return -1;
    }

    return 0;
}
void HWNvmediaIpcProducerContext::Attach(HWNvmediaOutputPipelineContext *io_poutputpipeline,int channel_id)
{
    NvSciStreamEventType event;
    NvSciError sciErr = NvSciError_Success;
    sciErr = NvSciStreamBlockEventQuery(io_poutputpipeline->_peventhandler->block_multicast, QUERY_TIMEOUT_FOREVER, &event);
    if (NvSciError_Success != sciErr || event != NvSciStreamEventType_SetupComplete) {
        PLOG_ERR("we can not attach consumer now.\n");
        return;
    }
    PLOG_DBG("ConnectLateConsumer, make sure multicast status is NvSciStreamEventType_SetupComplete before attach late consumer\n");

    std::string srcChannel = "nvscistream_" + std::to_string(io_poutputpipeline->psensorinfo->id * NUM_IPC_CONSUMERS * 2 + 2 * channel_id + 0);
    auto status = CFactory::CreateIpcBlock(_scisyncmodule, _scibufmodule, srcChannel.c_str(), true,
                        &io_poutputpipeline->_peventhandler->psrcIpcHandles[channel_id],
                        &io_poutputpipeline->_peventhandler->plateIpcEndpoint[channel_id]);
    if (status != SIPLStatus::NVSIPL_STATUS_OK) {
	    PLOG_DBG("HWNvmediaIpcProducerContext::Attach CFactory::CreateIpcBlock failed!\r\n");
	    return;
    }
    // PCHK_STATUS_AND_RETURN(status, "CFactory::Create ipc src Block");

    // connect late consumers to multicast
    sciErr = NvSciStreamBlockConnect(io_poutputpipeline->_peventhandler->block_multicast, io_poutputpipeline->_peventhandler->psrcIpcHandles[channel_id]);
    /* PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Multicast connect to ipc src"); */
    PLOG_DBG("Multicast is connected to ipc src: %u\n", channel_id);

    // indicate multicast block to proceed with the initialization and streaming with the connectting consumers
    sciErr = NvSciStreamBlockSetupStatusSet(io_poutputpipeline->_peventhandler->block_multicast, NvSciStreamSetup_Connect, true);
    /* PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Multicast set to NvSciStreamSetup_Connect!"); */

    // make sure relevant blocks reach streaming phase.
    // query consumers and queues
    sciErr = NvSciStreamBlockEventQuery(io_poutputpipeline->_peventhandler->psrcIpcHandles[channel_id], QUERY_TIMEOUT_FOREVER, &event);
    /* PCHK_NVSCICONNECT_AND_RETURN(sciErr, event, "Ipc src"); */
    PLOG_DBG("Ipc src: %u is connected.\n", channel_id);

    PLOG_DBG("Attach consumer success!\n");
    return;
}

void HWNvmediaIpcProducerContext::Detach(HWNvmediaOutputPipelineContext *io_poutputpipeline,int channel_id)
{
    NvSciError sciErr = NvSciStreamBlockDisconnect(io_poutputpipeline->_peventhandler->psrcIpcHandles[channel_id]);
    /* PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockDisconnect fail"); */
    if(NvSciError_Success != sciErr){
        PLOG_DBG("NvSciStreamBlockDisconnect fail!!!\n");
        return;
    }

    CFactory::ReleaseIpcBlock(io_poutputpipeline->_peventhandler->psrcIpcHandles[channel_id], io_poutputpipeline->_peventhandler->plateIpcEndpoint[channel_id]);
    io_poutputpipeline->_peventhandler->psrcIpcHandles[channel_id]   = 0U;
    io_poutputpipeline->_peventhandler->plateIpcEndpoint[channel_id] = 0U;

    PLOG_DBG("Detach consumer success!!!\n");
    return;
}

void HWNvmediaIpcProducerContext::AttachEventListen(HWNvmediaIpcProducerContext* context,HWNvmediaOutputPipelineContext *io_poutputpipeline)
{
   
    struct sensor_info o_sensor_info = { .sensor_id= static_cast<int>(io_poutputpipeline->psensorinfo->id), };

    set_producer_info(io_poutputpipeline->psensorinfo->id,&o_sensor_info);


    producer_start(io_poutputpipeline->psensorinfo->id);

    //listen server
    if (listen(io_poutputpipeline->_server_fd, 5) == -1) {
        cerr << "Failed to listen socket" << endl;
        close(io_poutputpipeline->_server_fd);
        return;
    }

    while (true) {
        // accept client
        struct sockaddr_un client_addr;
        socklen_t addr_len = sizeof(client_addr);
        int client_fd = accept(io_poutputpipeline->_server_fd, (struct sockaddr *)&client_addr, &addr_len);
        if (client_fd == -1) {
            cerr << "Failed to accept client connection" << endl;
            break;
        }
        std::thread clientEventThread(ProcessClientEventHandle,context, io_poutputpipeline, client_fd);
        clientEventThread.detach();
    }

    // close socket
    close(io_poutputpipeline->_server_fd);

}

void HWNvmediaIpcProducerContext::ProcessClientEventHandle(HWNvmediaIpcProducerContext* context,HWNvmediaOutputPipelineContext *io_poutputpipeline,int clientfd)
{
    char command[128];
    int read_len = -1;
    char idx[8];
    size_t o_idx = -1;

    int chinnal_id = -1;
    while(true){
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(clientfd, &readfds);
        struct timeval timeout;
        timeout.tv_sec = 2;  // set timeout 2s.
        timeout.tv_usec = 0;
        int ret = select(clientfd + 1, &readfds, NULL, NULL, &timeout);
        if(ret==0){//timeout
            continue;
        }
        if(ret<0){//exit
            break;
        }
        if(FD_ISSET(clientfd,&readfds)){
            read_len = recv(clientfd, command, sizeof(command), MSG_PEEK | MSG_DONTWAIT);
            if (read_len <= 0) {
                LOG_DBG("cuda client[%d] disconnect.\r\n",clientfd);
                break;
            }
            memset(command,0,128);
            read_len = recv(clientfd, command, sizeof(command), 0);
            if(read_len<0){
                continue;
            }
            command[read_len] = '\0';
            char *token;
            token = strtok(command, ":");
            if (token != NULL) {
                token = strtok(NULL, ":");
                if (token != NULL) {
                    chinnal_id = std::stoi(token);
                }
            }
            if (strcmp(command, "attach") == 0) {
                // exec attach
                context->Attach(io_poutputpipeline,chinnal_id);
            } else if (strcmp(command, "detach") == 0) {
                // exec detach
                context->Detach(io_poutputpipeline,chinnal_id);
                io_poutputpipeline->_lateAttach_idxManager->release_idx(chinnal_id);
                chinnal_id = -1;
            } else if(strcmp(command, "getidx")==0){
                // get free idx
                memset(idx,0,8);
                o_idx = io_poutputpipeline->_lateAttach_idxManager->get_free_idx();
                sprintf(idx,"%ld",o_idx);
                if(write(clientfd,idx,strlen(command)) == 0){
                    io_poutputpipeline->_lateAttach_idxManager->release_idx(o_idx);
                }
            }
        }
    }
    if(chinnal_id>=0){
        context->Detach(io_poutputpipeline,chinnal_id);
        io_poutputpipeline->_lateAttach_idxManager->release_idx(chinnal_id);
    }

    close(clientfd);
}
