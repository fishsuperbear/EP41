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
#include "sensor/nvs_consumer/CUtils.hpp"
#include "sensor/nvs_consumer/CChannel.hpp"
#include "nvscibuf.h"
#include "nvscisync.h"
#include "nvscistream.h"
#include "sensor/nvs_consumer/CIpcConsumerChannel.hpp"

#ifndef CMASTER_HPP
#define CMASTER_HPP

using namespace std;
using namespace nvsipl;

namespace hozon {
namespace netaos {
namespace desay { 

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

    SIPLStatus Setup(uint32_t multiNum)
    {
        // Camera Master setup
        m_upCamera = INvSIPLCamera::GetInstance();
        CHK_PTR_AND_RETURN(m_upCamera, "INvSIPLCamera::GetInstance()");

        auto sciErr = NvSciBufModuleOpen(&m_sciBufModule);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufModuleOpen");

        sciErr = NvSciSyncModuleOpen(&m_sciSyncModule);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncModuleOpen");

    
        sciErr = NvSciIpcInit();
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciIpcInit");

        multicastNum = multiNum;
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

    void StopStream(void)
    {
        for (auto i = 0U; i < MAX_NUM_SENSORS; i++) {
            if (nullptr != m_upChannels[i]) {
                m_upChannels[i]->Stop();
            }
        }
    }

    void DeinitPipeline(void)
    {
        auto status = m_upCamera->Deinit();
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("INvSIPLCamera::Deinit failed. status: %x\n", status);
        }
    }

    SIPLStatus RegisterSource(SensorInfo *pSensorInfo, CProfiler *pProfiler,uint32_t sensorIndex,uint32_t multicastIndex,const ConsumerConfig& consConfig)
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

        m_upChannels[pSensorInfo->id] = CreateChannel(pSensorInfo, pProfiler,sensorIndex*multicastNum,multicastIndex);
        CHK_PTR_AND_RETURN(m_upChannels[pSensorInfo->id], "Master CreateChannel");

        m_upChannels[pSensorInfo->id]->SetConsumerConfig(consConfig);

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

private:
    std::unique_ptr<CChannel> CreateChannel(SensorInfo *pSensorInfo, CProfiler *pProfiler,uint32_t channelStrIndex,uint32_t multicastIndex)
    {
        ConsumerType consumerType;
        auto status = GetConsumerTypeFromAppType(m_appType, consumerType);
        if (status != NVSIPL_STATUS_OK) {
            LOG_ERR("unexpected appType: %u\n", m_appType);
            return nullptr;
        } else {
            return std::unique_ptr<CIpcConsumerChannel>(
                new CIpcConsumerChannel(m_sciBufModule, m_sciSyncModule, pSensorInfo, consumerType,channelStrIndex,multicastIndex));
        }
    }

    AppType m_appType;
    unique_ptr<INvSIPLCamera> m_upCamera {nullptr};
    NvSciSyncModule m_sciSyncModule {nullptr};
    NvSciBufModule m_sciBufModule {nullptr};
    unique_ptr<CChannel> m_upChannels[MAX_NUM_SENSORS] {nullptr};
    uint32_t multicastNum = 1;
};

}
}
}

#endif //CMASTER_HPP
