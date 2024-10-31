/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/* STL Headers */
#include <cstring>
#include <iostream>
#include <unistd.h>

#include "CAppConfig.hpp"
#include "CChannel.hpp"
#include "CDisplayChannel.hpp"
#include "CIpcConsumerChannel.hpp"
#include "CIpcProducerChannel.hpp"
#include "COpenWFDController.hpp"
#include "CSingleProcessChannel.hpp"
#include "CUtils.hpp"
#include "NvSIPLCamera.hpp"
#include "NvSIPLPipelineMgr.hpp"

#include "nvscibuf.h"
#include "nvscistream.h"
#include "nvscisync.h"

#ifndef CMASTER_HPP
#define CMASTER_HPP

using namespace std;
using namespace nvsipl;

/** CMaster class */
class CMaster
{
  public:
    ~CMaster(void)
    {
        LOG_DBG("CMaster release.\n");

        // need to release other nvsci resources before closing modules.
        for (auto i = 0U; i < MAX_NUM_SENSORS; i++) {
            if (nullptr != m_upChannels[i]) {
                m_upChannels[i].reset();
            }
        }

        if (m_upDisplaychannel) {
            m_upDisplaychannel.reset();
        }

        if (m_sciBufModule != nullptr) {
            NvSciBufModuleClose(m_sciBufModule);
        }

        if (m_sciSyncModule != nullptr) {
            NvSciSyncModuleClose(m_sciSyncModule);
        }

        if (m_pAppConfig->GetCommType() != CommType_IntraProcess) {
            NvSciIpcDeinit();
        }
    }

    SIPLStatus Setup(CAppConfig *pAppConfig)
    {
        CHK_PTR_AND_RETURN(pAppConfig, "INvSIPLCamera::GetInstance()");
        m_pAppConfig = pAppConfig;

        // Camera Master setup
        m_upCamera = INvSIPLCamera::GetInstance();
        CHK_PTR_AND_RETURN(m_upCamera, "INvSIPLCamera::GetInstance()");

        auto sciErr = NvSciBufModuleOpen(&m_sciBufModule);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufModuleOpen");

        sciErr = NvSciSyncModuleOpen(&m_sciSyncModule);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncModuleOpen");

        if (m_pAppConfig->GetCommType() != CommType_IntraProcess) {
            sciErr = NvSciIpcInit();
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciIpcInit");
        }

        if (m_pAppConfig->IsDisplayEnabled()) {
            m_spWFDController = std::make_shared<COpenWFDController>();
            CHK_PTR_AND_RETURN(m_spWFDController, "COpenWFDController");
        }

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus SetPlatformConfig(PlatformCfg *pPlatformCfg, NvSIPLDeviceBlockQueues &queues)
    {
        return m_upCamera->SetPlatformCfg(pPlatformCfg, queues);
    }

    SIPLStatus
    SetPipelineConfig(uint32_t uIndex, NvSIPLPipelineConfiguration &pipelineCfg, NvSIPLPipelineQueues &pipelineQueues)
    {
        return m_upCamera->SetPipelineCfg(uIndex, pipelineCfg, pipelineQueues);
    }

    SIPLStatus InitPipeline()
    {
        auto status = m_upCamera->Init();
        CHK_STATUS_AND_RETURN(status, "m_upCamera->Init()");

        if (m_spWFDController) {
            status = m_spWFDController->InitResource();
            CHK_STATUS_AND_RETURN(status, " m_spWFDController->InitResource()");
        }

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus StartStream(void)
    {
        if (m_upDisplaychannel) {
            m_upDisplaychannel->Start();
        }

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

    SIPLStatus AttachConsumer()
    {
        if ((m_pAppConfig->GetCommType() != CommType_InterProcess) ||
            (m_pAppConfig->GetEntityType() != EntityType_Producer)) {
            LOG_WARN("Only IPC Producer support late attach by now\n");
            return NVSIPL_STATUS_ERROR;
        }

        for (auto i = 0U; i < MAX_NUM_SENSORS; i++) {
            if (nullptr != m_upChannels[i]) {
                CIpcProducerChannel *pIpcProducerChannel = dynamic_cast<CIpcProducerChannel *>(m_upChannels[i].get());
                pIpcProducerChannel->attach();
            }
        }
        return NVSIPL_STATUS_OK;
    }

    SIPLStatus DetachConsumer()
    {
        if ((m_pAppConfig->GetCommType() != CommType_InterProcess) ||
            (m_pAppConfig->GetEntityType() != EntityType_Producer)) {
            LOG_WARN("Only IPC Producer support late attach by now\n");
            return NVSIPL_STATUS_ERROR;
        }

        for (auto i = 0U; i < MAX_NUM_SENSORS; i++) {
            if (nullptr != m_upChannels[i]) {
                CIpcProducerChannel *pIpcProducerChannel = dynamic_cast<CIpcProducerChannel *>(m_upChannels[i].get());
                pIpcProducerChannel->detach();
            }
        }
        return NVSIPL_STATUS_OK;
    }

    void StopStream(void)
    {
        if (m_upDisplaychannel) {
            m_upDisplaychannel->Stop();
        }

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

        if (m_upDisplaychannel) {
            m_upDisplaychannel->Deinit();
        }

        for (auto i = 0U; i < MAX_NUM_SENSORS; i++) {
            if (nullptr != m_upChannels[i]) {
                m_upChannels[i]->Deinit();
            }
        }

        if (m_spWFDController) {
            m_spWFDController->DeInit();
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

        if (nullptr == m_upDisplaychannel && m_pAppConfig->IsDisplayEnabled()) {
            m_upDisplaychannel = CreateDisplayChannel(pSensorInfo, pProfiler);
            CHK_PTR_AND_RETURN(m_upDisplaychannel, "Display CreateChannel");

            auto status = m_upDisplaychannel->CreateBlocks(pProfiler);
            CHK_STATUS_AND_RETURN(status, "Display CreateBlocks");
        }

        m_upChannels[pSensorInfo->id] = CreateChannel(pSensorInfo, pProfiler, m_upDisplaychannel);
        CHK_PTR_AND_RETURN(m_upChannels[pSensorInfo->id], "Master CreateChannel");
        m_upChannels[pSensorInfo->id]->Init();

        auto status = m_upChannels[pSensorInfo->id]->CreateBlocks(pProfiler);
        CHK_STATUS_AND_RETURN(status, "Master CreateBlocks");

        return NVSIPL_STATUS_OK;
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

        if (m_upDisplaychannel) {
            auto status = InitStitchingToDisplay();
            CHK_STATUS_AND_RETURN(status, "InitStitchingToDisplay");
        }

        return NVSIPL_STATUS_OK;
    }

    SIPLStatus OnFrameAvailable(uint32_t uSensor, NvSIPLBuffers &siplBuffers)
    {
        if (uSensor >= MAX_NUM_SENSORS) {
            LOG_ERR("%s: Invalid sensor id: %u\n", __func__, uSensor);
            return NVSIPL_STATUS_BAD_ARGUMENT;
        }

        if (m_pAppConfig->GetCommType() == CommType_IntraProcess) {
            CSingleProcessChannel *pSingleProcessChannel =
                dynamic_cast<CSingleProcessChannel *>(m_upChannels[uSensor].get());
            return pSingleProcessChannel->Post(siplBuffers);
        } else if (m_pAppConfig->GetEntityType() == EntityType_Producer) {
            CIpcProducerChannel *pIpcProducerChannel = dynamic_cast<CIpcProducerChannel *>(m_upChannels[uSensor].get());
            return pIpcProducerChannel->Post(siplBuffers);
        } else {
            LOG_WARN("Received unexpected OnFrameAvailable, commType: %u, EntityType: %u\n",
                     m_pAppConfig->GetCommType(), m_pAppConfig->GetEntityType());
            return NVSIPL_STATUS_ERROR;
        }
    }

    SIPLStatus GetMaxErrorSize(const uint32_t devBlkIndex, size_t &size)
    {
        return m_upCamera->GetMaxErrorSize(devBlkIndex, size);
    }

    SIPLStatus GetErrorGPIOEventInfo(const uint32_t devBlkIndex, const uint32_t gpioIndex, SIPLGpioEvent &event)
    {
        return m_upCamera->GetErrorGPIOEventInfo(devBlkIndex, gpioIndex, event);
    }

    SIPLStatus GetDeserializerErrorInfo(const uint32_t devBlkIndex,
                                        SIPLErrorDetails *const deserializerErrorInfo,
                                        bool &isRemoteError,
                                        uint8_t &linkErrorMask)
    {
        return m_upCamera->GetDeserializerErrorInfo(devBlkIndex, deserializerErrorInfo, isRemoteError, linkErrorMask);
    }

    SIPLStatus GetModuleErrorInfo(const uint32_t index,
                                  SIPLErrorDetails *const serializerErrorInfo,
                                  SIPLErrorDetails *const sensorErrorInfo)
    {
        return m_upCamera->GetModuleErrorInfo(index, serializerErrorInfo, sensorErrorInfo);
    }

    SIPLStatus
    RegisterAutoControl(uint32_t uIndex, PluginType type, ISiplControlAuto *customPlugin, std::vector<uint8_t> &blob)
    {
        return m_upCamera->RegisterAutoControlPlugin(uIndex, type, customPlugin, blob);
    }

  private:
    std::unique_ptr<CChannel>
    CreateChannel(SensorInfo *pSensorInfo, CProfiler *pProfiler, std::unique_ptr<CDisplayChannel> &upDisplayChannel)
    {
        switch (m_pAppConfig->GetCommType()) {
            default:
            case CommType_IntraProcess:
                return make_unique<CSingleProcessChannel>(
                    m_sciBufModule, m_sciSyncModule, pSensorInfo, m_pAppConfig, m_upCamera.get(),
                    nullptr == upDisplayChannel ? nullptr : upDisplayChannel->GetDisplayProducer());
            case CommType_InterProcess:
                if (m_pAppConfig->GetEntityType() == EntityType_Producer) {
                    return make_unique<CP2pProducerChannel>(m_sciBufModule, m_sciSyncModule, pSensorInfo, m_pAppConfig,
                                                            m_upCamera.get());
                } else {
                    return make_unique<CP2pConsumerChannel>(m_sciBufModule, m_sciSyncModule, pSensorInfo, m_pAppConfig);
                }
            case CommType_InterChip:
                if (m_pAppConfig->GetEntityType() == EntityType_Producer) {
                    return make_unique<CC2cProducerChannel>(m_sciBufModule, m_sciSyncModule, pSensorInfo, m_pAppConfig,
                                                            m_upCamera.get());
                } else {
                    return make_unique<CC2cConsumerChannel>(m_sciBufModule, m_sciSyncModule, pSensorInfo, m_pAppConfig);
                }
        }
    }

    std::unique_ptr<CDisplayChannel> CreateDisplayChannel(SensorInfo *pSensorInfo, CProfiler *pProfiler)
    {
        return std::unique_ptr<CDisplayChannel>(
            new CDisplayChannel(m_sciBufModule, m_sciSyncModule, pSensorInfo, m_pAppConfig, m_spWFDController));
    }

    SIPLStatus InitStitchingToDisplay()
    {
        auto status = m_upDisplaychannel->Connect();
        CHK_STATUS_AND_RETURN(status, "CMaster: Display channel connect.");

        status = m_upDisplaychannel->InitBlocks();
        CHK_STATUS_AND_RETURN(status, "Display channel InitBlocks");

        status = m_upDisplaychannel->Reconcile();
        CHK_STATUS_AND_RETURN(status, "Display channel Reconcile");

        return NVSIPL_STATUS_OK;
    }

    CAppConfig *m_pAppConfig = nullptr;
    unique_ptr<INvSIPLCamera> m_upCamera{ nullptr };
    NvSciSyncModule m_sciSyncModule{ nullptr };
    NvSciBufModule m_sciBufModule{ nullptr };
    unique_ptr<CChannel> m_upChannels[MAX_NUM_SENSORS]{ nullptr };
    std::unique_ptr<CDisplayChannel> m_upDisplaychannel{ nullptr };
    std::shared_ptr<COpenWFDController> m_spWFDController{ nullptr };
};

#endif // CMASTER_HPP
