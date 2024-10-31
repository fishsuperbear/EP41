/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/* STL Headers */
#include <unistd.h>
#include <cstring>
#include <iostream>

#include "NvSIPLCamera.hpp"
#include "NvSIPLPipelineMgr.hpp"
#include "CUtils.hpp"
#include "CChannel.hpp"
#include "CSingleProcessChannel.hpp"
#include "CIpcProducerChannel.hpp"
#include "CIpcConsumerChannel.hpp"

#include "nvscibuf.h"
#include "nvscisync.h"
#include "nvscistream.h"

#ifndef CMASTER_HPP
#define CMASTER_HPP

using namespace std;
using namespace nvsipl;

/** CMaster class */
class CMaster
{
 public:
    CMaster(AppType appType):
        m_appType(appType)
    {
    }

    ~CMaster(void)
    {
        //need to release other nvsci resources before closing modules.
        for (auto i = 0U; i < MAX_NUM_SENSORS; i++) {
            if (nullptr != m_upChannels[i]) {
                m_upChannels[i].reset();
            }
        }

        LOG_DBG("CMaster release.\n");

        if (m_sciBufModule != nullptr) {
          NvSciBufModuleClose(m_sciBufModule);
        }

        if (m_sciSyncModule != nullptr) {
          NvSciSyncModuleClose(m_sciSyncModule);
        }
    }

    SIPLStatus Setup(bool bMultiProcess)
    {
        // Camera Master setup
        m_upCamera = INvSIPLCamera::GetInstance();
        CHK_PTR_AND_RETURN(m_upCamera, "INvSIPLCamera::GetInstance()");

        auto sciErr = NvSciBufModuleOpen(&m_sciBufModule);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufModuleOpen");

        sciErr = NvSciSyncModuleOpen(&m_sciSyncModule);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncModuleOpen");

        if (bMultiProcess) {
            sciErr = NvSciIpcInit();
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciIpcInit");
        }
        return NVSIPL_STATUS_OK;
    }

    SIPLStatus SetPlatformConfig(PlatformCfg* pPlatformCfg, NvSIPLDeviceBlockQueues &queues)
    {
        return m_upCamera->SetPlatformCfg(pPlatformCfg, queues);
    }

    SIPLStatus SetPipelineConfig(uint32_t uIndex, NvSIPLPipelineConfiguration &pipelineCfg, NvSIPLPipelineQueues &pipelineQueues)
    {
        return m_upCamera->SetPipelineCfg(uIndex, pipelineCfg, pipelineQueues);
    }

    SIPLStatus InitPipeline()
    {
        auto status = m_upCamera->Init();
        CHK_STATUS_AND_RETURN(status, "m_upCamera->Init()");

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus StartStream(void)
    {
        for (auto i = 0U; i < MAX_NUM_SENSORS; i++) {
            if (nullptr != m_upChannels[i]) {
                m_upChannels[i]->Start();
            }
        }

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus StartPipeline(void)
    {
        const SIPLStatus status = m_upCamera->Start();
        CHK_STATUS_AND_RETURN(status, "Start SIPL");
        return NVSIPL_STATUS_OK;
    }

    void StopStream(void)
    {
        for (auto i = 0U; i < MAX_NUM_SENSORS; i++) {
            if (nullptr != m_upChannels[i]) {
                m_upChannels[i]->Stop();
            }
        }
    }

    SIPLStatus StopPipeline(void)
    {
        const SIPLStatus status = m_upCamera->Stop();
        CHK_STATUS_AND_RETURN(status, "Stop SIPL");

        return NVSIPL_STATUS_OK;
    }

    void DeinitPipeline(void)
    {
        auto status = m_upCamera->Deinit();
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("INvSIPLCamera::Deinit failed. status: %x\n", status);
        }
    }

    SIPLStatus RegisterSource(SensorInfo *pSensorInfo, CProfiler *pProfiler)
    {
        LOG_DBG("CMaster: RegisterSource.\n");

        if (nullptr == pSensorInfo || nullptr == pProfiler) {
            LOG_ERR("%s: nullptr\n", __func__);
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }

        if (pSensorInfo->id >= MAX_NUM_SENSORS) {
            LOG_ERR("%s: Invalid sensor id: %u\n", __func__, pSensorInfo->id);
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }

        m_upChannels[pSensorInfo->id] = CreateChannel(pSensorInfo, pProfiler);
        CHK_PTR_AND_RETURN(m_upChannels[pSensorInfo->id], "Master CreateChannel");

        auto status = m_upChannels[pSensorInfo->id]->CreateBlocks(pProfiler);
        CHK_STATUS_AND_RETURN(status, "Master CreateBlocks");

        return NVSIPL_STATUS_OK;
    }

    void SetConsumerConfig(const ConsumerConfig& consConfig)
    {
        for (auto i = 0U; i < MAX_NUM_SENSORS; i++) {
            if (nullptr != m_upChannels[i]) {
                m_upChannels[i]->SetConsumerConfig(consConfig);
            }
        }
    }

    SIPLStatus InitStream(void)
    {
        LOG_DBG("CMaster: InitStream.\n");

        for (auto i = 0U; i < MAX_NUM_SENSORS; i++) {
            if (nullptr != m_upChannels[i]) {
                auto status = m_upChannels[i]->Connect();
                CHK_STATUS_AND_RETURN(status, "CMaster: Channel connect.");

                status = m_upChannels[i]->InitBlocks();
                CHK_STATUS_AND_RETURN(status, "InitBlocks");

                status = m_upChannels[i]->Reconcile();
                CHK_STATUS_AND_RETURN(status, "Channel Reconcile");
            }
        }

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus OnFrameAvailable(uint32_t uSensor, INvSIPLClient::INvSIPLBuffer *pBuffer)
    {
        if (uSensor >= MAX_NUM_SENSORS) {
            LOG_ERR("%s: Invalid sensor id: %u\n", __func__, uSensor);
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }

        if (m_appType == SINGLE_PROCESS) {
            CSingleProcessChannel* pSingleProcessChannel = dynamic_cast<CSingleProcessChannel*>(m_upChannels[uSensor].get());
            return pSingleProcessChannel->Post(pBuffer);
        } else if (m_appType == IPC_SIPL_PRODUCER) {
            CIpcProducerChannel* pIpcProducerChannel = dynamic_cast<CIpcProducerChannel*>(m_upChannels[uSensor].get());
            return pIpcProducerChannel->Post(pBuffer);
        } else {
            LOG_WARN("Received unexpected OnFrameAvailable, appType: %u\n", m_appType);
            return NVSIPL_STATUS_ERROR;
        }
    }

    SIPLStatus GetMaxErrorSize(const uint32_t devBlkIndex, size_t &size)
    {
        return m_upCamera->GetMaxErrorSize(devBlkIndex, size);
    }

    SIPLStatus GetErrorGPIOEventInfo(const uint32_t devBlkIndex,
                                     const uint32_t gpioIndex,
                                     SIPLGpioEvent &event)
    {
        return m_upCamera->GetErrorGPIOEventInfo(devBlkIndex, gpioIndex, event);
    }

    SIPLStatus GetDeserializerErrorInfo(const uint32_t devBlkIndex,
                                        SIPLErrorDetails * const deserializerErrorInfo,
                                        bool & isRemoteError,
                                        uint8_t& linkErrorMask)
    {
        return m_upCamera->GetDeserializerErrorInfo(devBlkIndex, deserializerErrorInfo,
                                                   isRemoteError, linkErrorMask);
    }

    SIPLStatus GetModuleErrorInfo(const uint32_t index,
                                         SIPLErrorDetails * const serializerErrorInfo,
                                         SIPLErrorDetails * const sensorErrorInfo)
    {
        return m_upCamera->GetModuleErrorInfo(index, serializerErrorInfo, sensorErrorInfo);
    }

    SIPLStatus RegisterAutoControl(uint32_t uIndex, PluginType type, ISiplControlAuto* customPlugin, std::vector<uint8_t>& blob)
    {
        return m_upCamera->RegisterAutoControlPlugin(uIndex, type, customPlugin, blob);
    }

private:
    std::unique_ptr<CChannel> CreateChannel(SensorInfo *pSensorInfo, CProfiler *pProfiler)
    {
        if (m_appType == SINGLE_PROCESS) {
            return std::unique_ptr<CSingleProcessChannel>(
                    new CSingleProcessChannel(m_sciBufModule, m_sciSyncModule, pSensorInfo, m_upCamera.get()));
        } else if (m_appType == IPC_SIPL_PRODUCER) {
            return std::unique_ptr<CIpcProducerChannel>(
                    new CIpcProducerChannel(m_sciBufModule, m_sciSyncModule, pSensorInfo, m_upCamera.get()));
        } else {
            ConsumerType consumerType;

            auto status = GetConsumerTypeFromAppType(m_appType, consumerType);
            if (status != NVSIPL_STATUS_OK) {
                LOG_ERR("unexpected appType: %u\n", m_appType);
                return nullptr;
            } else {
                return std::unique_ptr<CIpcConsumerChannel>(
                    new CIpcConsumerChannel(m_sciBufModule, m_sciSyncModule, pSensorInfo, consumerType));
            }
        }
    }

    AppType m_appType;
    unique_ptr<INvSIPLCamera> m_upCamera {nullptr};
    NvSciSyncModule m_sciSyncModule {nullptr};
    NvSciBufModule m_sciBufModule {nullptr};
    unique_ptr<CChannel> m_upChannels[MAX_NUM_SENSORS] {nullptr};
};

#endif //CMASTER_HPP
