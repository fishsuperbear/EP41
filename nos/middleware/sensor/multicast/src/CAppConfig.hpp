/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "Common.hpp"
#include "CUtils.hpp"

#ifndef CAPPCONFIG_HPP
#define CAPPCONFIG_HPP

using namespace std;
using namespace nvsipl;

class CAppConfig
{
  public:
    // clang-format off
    uint32_t GetVerbosity() const { return m_uVerbosity;}
#if !NV_IS_SAFETY
    const string& GetDynamicConfigName() const { return m_sDynamicConfigName; }
    const vector<uint32_t>& GetMasks() const { return m_vMasks; }
#endif
    const string& GetStaticConfigName() const { return m_sStaticConfigName; }
    const string& GetNitoFolderPath() const { return m_sNitoFolderPath; }
    CommType GetCommType() const { return m_eCommType; }
    EntityType GetEntityType() const { return m_eEntityType; }
    ConsumerType GetConsumerType() const { return m_eConsumerType; }
    QueueType GetQueueType() const { return m_eQueueType; }
    bool IsDisplayEnabled() const { return m_bEnableDisplay; }
    bool IsErrorIgnored() const { return m_bIgnoreError; }
    bool IsFileDumped() const { return m_bFileDump; }
    bool IsVersionShown() const { return m_bShowVersion; }
    bool IsMultiElementsEnabled() const { return m_bEnableMultiElements; }
    bool IsLateAttachEnabled() { return m_bEnableLateAttach; }
    uint8_t GetFrameFilter() const { return m_uFrameFilter; }
    uint32_t GetRunDurationSec() const { return m_uRunDurationSec; }
    PlatformCfg *GetPlatformCfg();

    SIPLStatus GetResolutionWidthAndHeight(uint32_t uSensorId, uint16_t &width, uint16_t &height);

    friend class CCmdLineParser;
    // clang-format on
  private:
    uint32_t m_uVerbosity = 1u;
#if !NV_IS_SAFETY
    string m_sDynamicConfigName;
    vector<uint32_t> m_vMasks;
#endif
    string m_sStaticConfigName;
    string m_sNitoFolderPath;
    CommType m_eCommType = CommType_IntraProcess;
    EntityType m_eEntityType = EntityType_Producer;
    ConsumerType m_eConsumerType = ConsumerType_Enc;
    QueueType m_eQueueType = QueueType_Fifo;
    uint8_t m_uFrameFilter = 1U;
    uint32_t m_uRunDurationSec = 0U;
    PlatformCfg m_platformCfg;

    // switch flag
    bool m_bEnableDisplay = false;
    bool m_bIgnoreError = false;
    bool m_bFileDump = false;
    bool m_bShowVersion = false;
    bool m_bEnableMultiElements = false;
    bool m_bEnableLateAttach = false;
};

#endif //CCMDPARSER_HPP
