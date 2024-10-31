/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "CPeerValidator.hpp"
#include <string>

/* Type of info */
#define VALIDATION_INFO_TYPE 0xabcd
/* Size of info */
#define VALIDATION_INFO_SIZE 256

static constexpr const char *INFO_IS_STATIC_CONFIG = "isStaticConfig";
static constexpr const char *INFO_PLATFORM_NAME = "platformName";
static constexpr const char *INFO_MASKS = "masks";
static constexpr const char *INFO_IS_MULTIELEMENTS_ENABLED = "isMultiElementsEnabled";

SIPLStatus CPeerValidator::SendValidationInfo(void)
{
    LOG_DBG("CPeerValidator::SendValidationInfo.\n");

    char stringInfo[VALIDATION_INFO_SIZE] = "";
    uint32_t infoSize = 0;

    GetStringInfo(infoSize, stringInfo);
    NvSciError err = NvSciStreamBlockUserInfoSet(m_handle, VALIDATION_INFO_TYPE, infoSize, stringInfo);
    CHK_NVSCISTATUS_AND_RETURN(err, "failed to send validation info.");
    return NVSIPL_STATUS_OK;
}

SIPLStatus CPeerValidator::Validate(void)
{
    LOG_DBG("CPeerValidator::Validate.\n");

    uint32_t size = VALIDATION_INFO_SIZE;
    char infoProd[VALIDATION_INFO_SIZE] = "";
    SIPLStatus status = NVSIPL_STATUS_OK;

    NvSciError err = NvSciStreamBlockUserInfoGet(m_handle, NvSciStreamBlockType_Producer, 0U, VALIDATION_INFO_TYPE,
                                                 &size, &infoProd);
    if (NvSciError_Success == err) {
        char stringInfo[VALIDATION_INFO_SIZE] = "";
        uint32_t infoSize = 0;
        GetStringInfo(infoSize, stringInfo);
        LOG_INFO("the info from producer: %s.", infoProd);
        LOG_INFO("the info in this consumer: %s.", stringInfo);
        status = (0 == std::strcmp(stringInfo, infoProd)) ? NVSIPL_STATUS_OK : NVSIPL_STATUS_ERROR;
    } else if (NvSciError_StreamInfoNotProvided == err) {
        LOG_ERR("validation info not provided by the producer.\n");
        status = NVSIPL_STATUS_ERROR;
    } else {
        LOG_ERR("failed to query the producer info.\n");
        status = NVSIPL_STATUS_ERROR;
    }

    return status;
}

void CPeerValidator::GetStringInfo(uint32_t &size, char *outputInfo)
{
    LOG_DBG("CPeerValidator::GetStringInfo\n");

    int isStaticConfig;
    std::string platformName;
    std::string masks;
    int isMultiElementsEnabled = (int)m_pAppConfig->IsMultiElementsEnabled();

#if !NV_IS_SAFETY
    if (!m_pAppConfig->GetDynamicConfigName().empty()) {
        isStaticConfig = 0;
        platformName = m_pAppConfig->GetDynamicConfigName();
    } else {
#endif
        isStaticConfig = 1;
        platformName = m_pAppConfig->GetStaticConfigName().empty() ? "F008A120RM0AV2_CPHY_x4"
                                                                   : m_pAppConfig->GetStaticConfigName();
#if !NV_IS_SAFETY
    }
    for (auto m : m_pAppConfig->GetMasks()) {
        masks += to_string(m) + ",";
    }
#endif
    size = snprintf(outputInfo, VALIDATION_INFO_SIZE, "%s=%d;%s=%s;%s=%s;%s=%d;", INFO_IS_STATIC_CONFIG,
                    isStaticConfig, INFO_PLATFORM_NAME, platformName.c_str(), INFO_MASKS, masks.c_str(),
                    INFO_IS_MULTIELEMENTS_ENABLED, isMultiElementsEnabled);
}