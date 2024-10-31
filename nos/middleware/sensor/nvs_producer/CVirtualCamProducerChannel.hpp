// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#ifndef MIDDLEWARE_SENSOR_NVS_PRODUCER_CVIRTUALCAMPRODUCERCHANNEL_HPP_
#define MIDDLEWARE_SENSOR_NVS_PRODUCER_CVIRTUALCAMPRODUCERCHANNEL_HPP_

#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "CChannel.hpp"
#include "CFactory.hpp"
#include "CPoolManager.hpp"
#include "CVirtualCamProducer.hpp"
#include "CVirtualCamVicProducer.hpp"
#include "atomic_queue.h"

using namespace nvsipl;
using hozon::netaos::codec::InputBufPtr;
using hozon::netaos::codec::PicInfos;

#define MAX_NUM_CONSUMERS_EX 1

class CVirtualCamProducerChannel : public CChannel {
   public:
    CVirtualCamProducerChannel() = delete;

    CVirtualCamProducerChannel(NvSciBufModule& bufMod, NvSciSyncModule& syncMod, PicInfo* pic_info,
                               SensorInfo* pSensorInfo)  //
        : CChannel("VirtualCamProducerChannel", bufMod, syncMod, pSensorInfo) {
        pic_info_ = pic_info;
        for (auto i = 0U; i < NUM_IPC_CONSUMERS; i++) {
            // m_srcChannels[i] = "nvscistream_" + std::to_string(m_pSensorInfo->id * MAX_NUM_CONSUMERS_EX * 2 + 2 * i + 0);
            m_srcChannels[i] = std::string("cam") + std::to_string(pSensorInfo->id) + "_send" + std::to_string(i);
            m_srcIpcHandles[i] = 0U;
            PLOG_ERR("CVirtualCamProducerChannel=%s.\n", m_srcChannels[i].data());
        }
    }

    ~CVirtualCamProducerChannel(void) {
        PLOG_DBG("Release.\n");

        if (m_upPoolManager != nullptr && m_upPoolManager->GetHandle() != 0U) {
            (void)NvSciStreamBlockDelete(m_upPoolManager->GetHandle());
        }
        if (m_multicastHandle != 0U) {
            (void)NvSciStreamBlockDelete(m_multicastHandle);
        }
        if (m_upProducer != nullptr && m_upProducer->GetHandle() != 0U) {
            (void)NvSciStreamBlockDelete(m_upProducer->GetHandle());
        }
        for (auto i = 0U; i < NUM_IPC_CONSUMERS; i++) {
            if (m_rtnSyncHandles[i] != 0U) {
                (void)NvSciStreamBlockDelete(m_rtnSyncHandles[i]);
            }
            if (m_limiterHandles[i] != 0U) {
                (void)NvSciStreamBlockDelete(m_limiterHandles[i]);
            }
            if (m_srcIpcHandles[i] != 0U) {
                (void)NvSciStreamBlockDelete(m_srcIpcHandles[i]);
            }
            if (m_srcIpcEndpoints[i]) {
                (void)NvSciIpcCloseEndpointSafe(m_srcIpcEndpoints[i], false);
            }
        }
    }

    virtual SIPLStatus Init(void) {
        lateConsumerHelper = std::make_shared<CLateConsumerHelper>(m_bufModule, m_syncModule);
        m_uEarlyConsCount = NUM_IPC_CONSUMERS - lateConsumerHelper->GetLateConsCount();
        return NVSIPL_STATUS_OK;
    }

    virtual SIPLStatus Deinit(void) {
        is_stop_ = true;
        if (post_thread_ && post_thread_->joinable()) {
            post_thread_->join();
            post_thread_.reset();
        }

        if (m_upProducer != nullptr) {
            m_upProducer->Deinit();
        }
        printf("[%d] deinit done.\n", m_pSensorInfo->id);

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus Post(InputBufPtr buf) {
        PLOG_DBG("Post\n");
        // block if queue is full.
        if (buf->data.size()) {
            while (!decoded_queue_.try_push(buf)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            };
        } else {
            PLOG_ERR("post buffer is null!\n");
            return NVSIPL_STATUS_ERROR;
        }

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus CreateBlocks(CProfiler* pProfiler) {
        PLOG_DBG("CreateBlocks.\n");

        m_upPoolManager = CFactory::CreatePoolManager(m_pSensorInfo->id, MAX_NUM_PACKETS);
        PCHK_PTR_AND_RETURN(m_upPoolManager, "CFactory::CreatePoolManager");
        PLOG_DBG("PoolManager is created.\n");
        m_upPoolManager->PreInit(lateConsumerHelper);
        // TODO(mxt): config later.
        if (m_pSensorInfo->id == 0 || m_pSensorInfo->id == 1) {
            m_upProducer = CreateVirtualCamProducer(m_upPoolManager->GetHandle(), pic_info_);
            PCHK_PTR_AND_RETURN(m_upProducer, "CFactory::CreateVirtualCamProducer");
            auto producer = dynamic_cast<CVirtualCamProducer*>(m_upProducer.get());
            producer->PreInit(lateConsumerHelper);
        } else {
            m_upProducer = CreateVirtualCamVicProducer(m_upPoolManager->GetHandle(), pic_info_);
            PCHK_PTR_AND_RETURN(m_upProducer, "CFactory::CreateVirtualCamVicProducer");
            auto producer = dynamic_cast<CVirtualCamVicProducer*>(m_upProducer.get());
            producer->PreInit(lateConsumerHelper);
        }
        m_upProducer->SetProfiler(pProfiler);
        PLOG_DBG("Producer is created.\n");

        if (NUM_IPC_CONSUMERS > 1) {
            auto status = CFactory::CreateMulticastBlock(NUM_IPC_CONSUMERS, m_multicastHandle);
            PCHK_STATUS_AND_RETURN(status, "CFactory::CreateMulticastBlock");
            PLOG_DBG("Multicast block is created.\n");
        }

        for (auto i = 0U; i < m_uEarlyConsCount; i++) {
            auto status = CreateIpcSrcAndEndpoint(GetSrcChannel(m_pSensorInfo->id, i), &m_srcIpcEndpoints[i],
                                                  &m_srcIpcHandles[i]);
            PCHK_STATUS_AND_RETURN(status, "Create ipc src block");
            PLOG_DBG("Ipc src block: %u is created, srcChannel: %s.\n", i, GetSrcChannel(m_pSensorInfo->id, i).c_str());
            status = CFactory::CreateReturnSyncBlock(m_syncModule, m_rtnSyncHandles[i]);
            PCHK_STATUS_AND_RETURN(status, "CFactory::CreateReturnSyncBlock");
            PLOG_DBG("ReturnSync block is created.\n");
            status = CFactory::CreateLimiterBlock(MAX_LIMITER_COUNT, m_limiterHandles[i]);
            PCHK_STATUS_AND_RETURN(status, "CFactory::CreateLimiterBlock");
            PLOG_DBG("Limiter block is created.\n");
        }

        return NVSIPL_STATUS_OK;
    }

    virtual SIPLStatus Connect(void) {
        NvSciStreamEventType event;
        NvSciStreamBlock producerLink = 0U;
        NvSciStreamBlock consumerLink = 0U;
        NvSciError sciErr = NvSciError_Success;

        PLOG_DBG("Connect.\n");

        if (NUM_IPC_CONSUMERS == 1U) {
            producerLink = m_upProducer->GetHandle();
        } else {
            // connect producer with multicast
            sciErr = NvSciStreamBlockConnect(m_upProducer->GetHandle(), m_multicastHandle);
            PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Connect producer to multicast");
            PLOG_DBG("Producer is connected to multicast.\n");
            producerLink = m_multicastHandle;
        }
        for (auto i = 0U; i < m_uEarlyConsCount; i++) {
            vector<NvSciStreamBlock> vConsumerLinks{};
            PopulateConsumerLinks(i, vConsumerLinks);
            if (vConsumerLinks.size() > 0) {
                consumerLink = vConsumerLinks[0];
                //for C2C
                for (auto j = 1U; j < vConsumerLinks.size(); j++) {
                    sciErr = NvSciStreamBlockConnect(vConsumerLinks[j], consumerLink);
                    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "connect consumer link");
                    consumerLink = vConsumerLinks[j];
                }
#if 1
                sciErr = NvSciStreamBlockConnect(m_rtnSyncHandles[i], consumerLink);
                PCHK_NVSCISTATUS_AND_RETURN(sciErr, "RtnSync connect to ipc src");
                sciErr = NvSciStreamBlockConnect(m_limiterHandles[i], m_rtnSyncHandles[i]);
                PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Limiter connect to ipc src");
                PLOG_DBG("Limiter is connected to ipc src: %u\n", i);
                sciErr = NvSciStreamBlockConnect(producerLink, m_limiterHandles[i]);
                PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Multicast connect to ipc src");
                PLOG_DBG("Multicast is connected to ipc src: %u\n", i);
#else
                sciErr = NvSciStreamBlockConnect(producerLink, consumerLink);
                PCHK_NVSCISTATUS_AND_RETURN(sciErr, "producerLink connects to consumerLink");
                PLOG_DBG("producerLink is connected to consumerLink: %u\n", i);
#endif
            }
        }

        // indicate Multicast to proceed with the initialization and streaming with the connected consumers
        if (m_multicastHandle && lateConsumerHelper && lateConsumerHelper->GetLateConsCount()) {
            sciErr = NvSciStreamBlockSetupStatusSet(m_multicastHandle, NvSciStreamSetup_Connect, true);
            PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Multicast status set to NvSciStreamSetup_Connect");
        }

        LOG_MSG("Producer is connecting to the stream...\n");
        vector<NvSciStreamBlock> vQueryBlocks{};
        PopulateQueryBlocks(vQueryBlocks);
        if (vQueryBlocks.size() > 0) {
            for (auto i = 0U; i < vQueryBlocks.size(); i++) {
                sciErr = NvSciStreamBlockEventQuery(vQueryBlocks[i], QUERY_TIMEOUT_FOREVER, &event);
                PCHK_NVSCICONNECT_AND_RETURN(sciErr, event, "query block");
                PLOG_DBG("Query block: %u is connected.\n", i);
            }
        }

        LOG_MSG("Producer is connected to the stream!\n");
        return NVSIPL_STATUS_OK;
    }

    virtual SIPLStatus Reconcile(void) {
        SIPLStatus status = CChannel::Reconcile();

        NvSciStreamEventType event;
        // query multicast
        if (m_multicastHandle != 0U) {
            NvSciError sciErr = NvSciStreamBlockEventQuery(m_multicastHandle, QUERY_TIMEOUT_FOREVER, &event);
            if (NvSciError_Success != sciErr || event != NvSciStreamEventType_SetupComplete) {
                PLOG_ERR("Multicast block setup error: 0x%.8X .\n", sciErr);
                return NVSIPL_STATUS_ERROR;
            }
            PLOG_DBG("Multicast block is setup complete.\n");
            // m_isReadyForLateAttach = true;
        }

        return status;
    }

    SIPLStatus attach(uint32_t i_lateIdx) {
        // if (m_isLateConsumerAttached) {
        //     PLOG_WARN("Late consumer is already attached!");
        //     return NVSIPL_STATUS_OK;
        // }

        // if (!m_isReadyForLateAttach) {
        //     PLOG_WARN("It is NOT ready for attach now!");
        //     return NVSIPL_STATUS_OK;
        // }
        // m_isReadyForLateAttach = false;

        NvSciStreamEventType event;
        NvSciError sciErr = NvSciError_Success;
        PLOG_DBG(
            "ConnectLateConsumer, make sure multicast status is NvSciStreamEventType_SetupComplete before attach "
            "late consumer\n");

        // create ipcsrc block for late consumer
        // for (auto i = lateIdx; i < NUM_IPC_CONSUMERS; i++) {
        auto status = CreateIpcSrcAndEndpoint(GetSrcChannel(m_pSensorInfo->id, i_lateIdx), &m_lateIpcEndpoint,
                                              &m_srcIpcHandles[i_lateIdx]);
        PCHK_STATUS_AND_RETURN(status, "CFactory::Create ipc src Block");
        // }

        // connect late consumers to multicast
        // for (auto i = lateIdx; i < NUM_IPC_CONSUMERS; i++) {
        sciErr = NvSciStreamBlockConnect(m_multicastHandle, m_srcIpcHandles[i_lateIdx]);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Multicast connect to ipc src");
        PLOG_DBG("Multicast is connected to ipc src: %u\n", i_lateIdx);
        // }

        // indicate multicast block to proceed with the initialization and streaming with the connectting consumers
        sciErr = NvSciStreamBlockSetupStatusSet(m_multicastHandle, NvSciStreamSetup_Connect, true);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Multicast set to NvSciStreamSetup_Connect!");

        // make sure relevant blocks reach streaming phase.
        // query consumers and queues
        // for (auto i = lateIdx; i < NUM_IPC_CONSUMERS; i++) {
        sciErr = NvSciStreamBlockEventQuery(m_srcIpcHandles[i_lateIdx], QUERY_TIMEOUT_FOREVER, &event);
        PCHK_NVSCICONNECT_AND_RETURN(sciErr, event, "Ipc src");
        PLOG_DBG("Ipc src: %u is connected.\n", i_lateIdx);
        // }

        // make sure late consumers attach success
        sciErr = NvSciStreamBlockEventQuery(m_multicastHandle, QUERY_TIMEOUT, &event);
        if (NvSciError_Success != sciErr || event != NvSciStreamEventType_SetupComplete) {
            PLOG_ERR("Multicast block setup error: 0x%.8X event=%d.\n", sciErr, event);

            // release relevant resource if attach fail
            CFactory::ReleaseIpcBlock(m_lateIpcEndpoint, m_srcIpcHandles[i_lateIdx]);
            m_srcIpcHandles[i_lateIdx] = 0U;
            m_lateIpcEndpoint = 0U;

            // reset this flag to allow next attach
            m_isReadyForLateAttach = true;
            return NVSIPL_STATUS_ERROR;
        }
        m_isReadyForLateAttach = true;
        m_isLateConsumerAttached = true;

        LOG_MSG("Late consumer is attached successfully!\n");
        return NVSIPL_STATUS_OK;
    }

    SIPLStatus detach(uint32_t i_lateIdx) {
        if (!m_isLateConsumerAttached) {
            PLOG_WARN("Late consumer is already detached!");
            return NVSIPL_STATUS_OK;
        }

        // uint32_t lateIdx = i_lateIdx;
        NvSciError sciErr = NvSciStreamBlockDisconnect(m_srcIpcHandles[i_lateIdx]);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockDisconnect fail");
        m_isLateConsumerAttached = false;

        CFactory::ReleaseIpcBlock(m_lateIpcEndpoint, m_srcIpcHandles[i_lateIdx]);
        m_srcIpcHandles[i_lateIdx] = 0U;
        m_lateIpcEndpoint = 0U;

        LOG_MSG("Late consumer is detached successfully!\n");
        return NVSIPL_STATUS_OK;
    }

    virtual SIPLStatus InitBlocks(void) {
        PLOG_DBG(": InitBlocks.\n");

        auto status = m_upPoolManager->Init();
        PCHK_STATUS_AND_RETURN(status, "Pool Init");

        status = m_upProducer->Init(m_bufModule, m_syncModule);
        PCHK_STATUS_AND_RETURN(status, (m_upProducer->GetName() + " Init").c_str());

        return NVSIPL_STATUS_OK;
    }

   protected:
    virtual const string GetSrcChannel(uint32_t sensorId, uint32_t consumerId) const {
        //return IPC_CHANNEL_PREFIX + std::to_string(sensorId * NUM_IPC_CONSUMERS * 2 + 2 * consumerId + 0);
        return m_srcChannels[consumerId];
    }

    virtual SIPLStatus CreateIpcSrcAndEndpoint(const string& srcChannel, NvSciIpcEndpoint* pEndPoint,
                                               NvSciStreamBlock* pIpcSrc) {
        auto status = CFactory::CreateIpcBlock(m_syncModule, m_bufModule, srcChannel.c_str(), true, pEndPoint, pIpcSrc);
        PCHK_STATUS_AND_RETURN(status, "CFactory create ipc src block");

        return NVSIPL_STATUS_OK;
    }

    virtual void PopulateConsumerLinks(uint32_t consumerId, vector<NvSciStreamBlock>& vConsumerLinks) {
        vConsumerLinks.push_back(m_srcIpcHandles[consumerId]);
    };

    virtual void PopulateQueryBlocks(vector<NvSciStreamBlock>& vQueryBlocks) {
        vQueryBlocks.push_back(m_upProducer->GetHandle());
        vQueryBlocks.push_back(m_upPoolManager->GetHandle());
        for (auto i = 0U; i < m_uEarlyConsCount; i++) {
            vQueryBlocks.push_back(m_srcIpcHandles[i]);
        }
        if (m_multicastHandle != 0U) {
            vQueryBlocks.push_back(m_multicastHandle);
        }
    };

    virtual void GetEventThreadHandlers(bool isStreamRunning, std::vector<CEventHandler*>& vEventHandlers) {
        if (!isStreamRunning) {
            vEventHandlers.push_back(m_upPoolManager.get());
        } else {
            // TODO(mxt): start after setup step.
            post_thread_.reset(new std::thread(&CVirtualCamProducerChannel::PostThread, this));
        }
        vEventHandlers.push_back(m_upProducer.get());
    }

    std::unique_ptr<CProducer> CreateVirtualCamProducer(NvSciStreamBlock poolHandle, PicInfo* pic_info) {
        NvSciStreamBlock producerHandle = 0U;
        unique_ptr<CProducer> upProducer = nullptr;

        auto sciErr = NvSciStreamProducerCreate(poolHandle, &producerHandle);
        if (sciErr != NvSciError_Success) {
            LOG_ERR("CreateVirtualCamProducer failed: 0x%x.\n", sciErr);
            return nullptr;
        }
        upProducer.reset(new CVirtualCamProducer(producerHandle, pic_info));
        std::vector<ElementInfo> elemsInfo{
            {ELEMENT_TYPE_ICP_RAW, true}, {ELEMENT_TYPE_NV12_BL, true}, {ELEMENT_TYPE_METADATA, true}};

        upProducer->SetPacketElementsInfo(elemsInfo);
        return upProducer;
    }

    std::unique_ptr<CProducer> CreateVirtualCamVicProducer(NvSciStreamBlock poolHandle, PicInfo* pic_info) {
        NvSciStreamBlock producerHandle = 0U;
        unique_ptr<CProducer> upProducer = nullptr;

        auto sciErr = NvSciStreamProducerCreate(poolHandle, &producerHandle);
        if (sciErr != NvSciError_Success) {
            LOG_ERR("CreateVirtualCamProducer failed: 0x%x.\n", sciErr);
            return nullptr;
        }
        upProducer.reset(new CVirtualCamVicProducer(producerHandle, pic_info));
        std::vector<ElementInfo> elemsInfo{
            {ELEMENT_TYPE_ICP_RAW, true}, {ELEMENT_TYPE_NV12_BL, false}, {ELEMENT_TYPE_METADATA, true}};

        upProducer->SetPacketElementsInfo(elemsInfo);
        return upProducer;
    }

    bool IsProducerReady() {
        if (pic_info_->sid == 0 || pic_info_->sid == 1) {
            auto producer = (CVirtualCamProducer*)m_upProducer.get();
            return producer->IsComplete();

        } else {
            auto producer = (CVirtualCamVicProducer*)m_upProducer.get();
            return producer->IsComplete();
        }
    }

    void PostThread() {
        while (!is_stop_) {
            while (!is_stop_ && !IsProducerReady()) {
                LOG_INFO("sid=%d wait....\n", m_pSensorInfo->id);
                printf("sid=%d wait....\n", m_pSensorInfo->id);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            // will block if queue is empty.
            InputBufPtr buf;
            if (!decoded_queue_.try_pop(buf)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            };
            // LOG_ERR("get one frame, size=%d\n", buf->data.size());
            if (buf) {
                auto status = m_upProducer->Post((void*)buf.get());
            }

            // auto now = std::chrono::high_resolution_clock::now();
            // auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch());

            // if ((uint64_t)duration.count() < buf->post_time) {
            //     auto wait = std::chrono::nanoseconds(buf->post_time) - duration;
            //     PLOG_ERR("[%d]===wait %ums\n", pic_info_->sid, wait.count() / 1000000);
            //     std::this_thread::sleep_for(wait);
            // } else {
            //     PLOG_ERR("post_time before now!  %lu/%lu/%ld\n", buf->post_time, duration.count(), (int64_t)(duration.count() - buf->post_time));
            //     // buffer_pool_->ReleaseBuffer(packetIndex);
            //     // return NVSIPL_STATUS_ERROR;
            // }
        }
    }

    std::unique_ptr<CProducer> m_upProducer = nullptr;

   private:
    unique_ptr<CPoolManager> m_upPoolManager = nullptr;
    NvSciStreamBlock m_multicastHandle = 0U;
    NvSciStreamBlock m_limiterHandles[MAX_NUM_CONSUMERS];
    NvSciStreamBlock m_rtnSyncHandles[MAX_NUM_CONSUMERS];
    NvSciStreamBlock m_srcIpcHandles[NUM_IPC_CONSUMERS]{};
    NvSciIpcEndpoint m_srcIpcEndpoints[NUM_IPC_CONSUMERS]{};

    string m_srcChannels[MAX_NUM_CONSUMERS];
    uint32_t m_uMulticastNum;
    bool m_isLateConsumerAttached = false;
    bool m_isReadyForLateAttach = false;
    uint32_t m_uEarlyConsCount = NUM_IPC_CONSUMERS;
    NvSciIpcEndpoint m_lateIpcEndpoint = 0U;
    std::shared_ptr<CLateConsumerHelper> lateConsumerHelper = nullptr;

    PicInfo* pic_info_;
    using AtomicQueue = atomic_queue::AtomicQueue2<InputBufPtr, 1, false, false, false, true>;
    AtomicQueue decoded_queue_;
    std::unique_ptr<std::thread> post_thread_;
    std::mutex mtx_;
    std::atomic_bool is_stop_{false};
};

#endif  // MIDDLEWARE_SENSOR_NVS_PRODUCER_CVIRTUALCAMPRODUCERCHANNEL_HPP_
