/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "CAppConfig.hpp"
#if !NV_IS_SAFETY
#include "NvSIPLQuery.hpp"      // Query
#include "NvSIPLQueryTrace.hpp" // Query Trace
#endif
#include "platform/ar0820.hpp"

PlatformCfg *CAppConfig::GetPlatformCfg()
{
    if (m_platformCfg.numDeviceBlocks == 0U) {
#if !NV_IS_SAFETY
        if (m_sDynamicConfigName != "") {
            // INvSIPLQuery
            auto pQuery = INvSIPLQuery::GetInstance();
            if (pQuery == nullptr) {
                LOG_ERR("INvSIPLQuery::GetInstance() return null.\n");
                return nullptr;
            }

            auto status = pQuery->ParseDatabase();
            if (status != NVSIPL_STATUS_OK) {
                LOG_ERR("INvSIPLQuery::ParseDatabase() failed.\n");
                return nullptr;
            }

            status = pQuery->GetPlatformCfg(m_sDynamicConfigName, m_platformCfg);
            if (status != NVSIPL_STATUS_OK) {
                LOG_ERR("INvSIPLQuery::GetPlatformCfg failed, status: %u\n", status);
                return nullptr;
            }
            // Apply mask
            LOG_INFO("Setting link masks\n");
            status = pQuery->ApplyMask(m_platformCfg, m_vMasks);
            if (status != NVSIPL_STATUS_OK) {
                LOG_ERR("INvSIPLQuery::ApplyMask failed, status: %u\n", status);
                return nullptr;
            }
        } else {
#endif // !NV_IS_SAFETY
            if (m_sStaticConfigName == "" || m_sStaticConfigName == "F008A120RM0AV2_CPHY_x4") {
                m_platformCfg = platformCfgAr0820;
            } else {
                LOG_ERR("Unexpected platform configuration\n");
                return nullptr;
            }
#if !NV_IS_SAFETY
        }
#endif // !NV_IS_SAFETY
    }

    return &m_platformCfg;
}

SIPLStatus CAppConfig::GetResolutionWidthAndHeight(uint32_t uSensorId, uint16_t &width, uint16_t &height)
{
    for (auto d = 0u; d != m_platformCfg.numDeviceBlocks; d++) {
        auto db = m_platformCfg.deviceBlockList[d];
        for (auto m = 0u; m != db.numCameraModules; m++) {
            SensorInfo *pSensorInfo = &db.cameraModuleInfoList[m].sensorInfo;
            if (pSensorInfo->id == uSensorId) {
                width = (uint16_t)pSensorInfo->vcInfo.resolution.width;
                height = (uint16_t)pSensorInfo->vcInfo.resolution.height;

                return NVSIPL_STATUS_OK;
            }
        }
    }

    LOG_ERR("CAppConfig::GetResolutionWidthAndHeight failed. uSensorId: %u\n", uSensorId);
    return NVSIPL_STATUS_ERROR;
}
