/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef CPEERVALIDATIOR_H
#define CPEERVALIDATIOR_H

#include <unistd.h>
#include "CAppConfig.hpp"
#include "CUtils.hpp"
#include "NvSIPLCommon.hpp"

#include "nvscistream.h"

class CPeerValidator
{
  public:
    CPeerValidator() = default;
    explicit CPeerValidator(CAppConfig *pAppConfig)
        : m_pAppConfig(pAppConfig)
    {
    }

    inline void SetHandle(const NvSciStreamBlock &handle)
    {
        LOG_DBG("CPeerValidator::SetHandle.\n");

        m_handle = handle;
    }

    inline NvSciStreamBlock GetHandle(void) const
    {
        LOG_DBG("CPeerValidator::GetHandle.\n");

        return m_handle;
    }

    inline SIPLStatus SetAppConfig(CAppConfig *pAppConfig)
    {
        LOG_DBG("CPeerValidator::SetAppConfig.\n");

        CHK_PTR_AND_RETURN(pAppConfig, "CPeerValidator::SetAppConfig");
        m_pAppConfig = pAppConfig;
        return NVSIPL_STATUS_OK;
    }

    SIPLStatus SendValidationInfo(void);
    SIPLStatus Validate(void);

  private:
    void GetStringInfo(uint32_t &size, char *outputInfo);

    NvSciStreamBlock m_handle{ 0U };
    const CAppConfig *m_pAppConfig{ nullptr };
};

#endif