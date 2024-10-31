/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: rtf event latency API header
 * Create: 2020-12-22
 * Notes: N/A
 */

#ifndef RTFTOOLS_RTFEVENT_LATENCY_H
#define RTFTOOLS_RTFEVENT_LATENCY_H

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "ara/core/map.h"
#include "ara/core/string.h"
#include "rtf/com/rtf_com.h"
#include "rtf/internal/tools_common_client_manager.h"
#include "rtf/rtfevent/RtfEvent.h"

namespace rtf {
namespace rtfevent {
using LatencyResult = rtf::maintaind::LatencyResult;
using LatencyResultMap = ara::core::Map<ara::core::String, rtf::rtfevent::LatencyResult>;
using LatencyMode = rtf::maintaind::LatencyMode;
struct EventLatencyStatus {
    ara::core::String pubApplicationName;
    rtf::rtfevent::LatencyMode pubStatus;
    ara::core::Map<ara::core::String, rtf::rtfevent::LatencyMode> subStatus;
};
class RtfEventLatency {
public:
RtfEventLatency();
~RtfEventLatency();
int Init();
int EnableLatencyMode(const ara::core::String& eventName);
int QueryLatencyStatus(const ara::core::String& eventName, rtf::rtfevent::EventLatencyStatus& status);
int QueryLatencyResult(
        const ara::core::String& eventName, const std::uint16_t& timeWindows, LatencyResultMap& resultMap);

int DisableLatencyMode(const ara::core::String& eventName);

private:
int SwitchLatencyMode(const ara::core::String& eventName, const rtf::maintaind::LatencyMode& mode);
void AddEventLatencyStatus(EventLatencyStatus& status, const rtf::maintaind::LatencyStatus& latencyStatus,
                           const bool& isOnline) const;
void GetInfoResult(const rtf::maintaind::proxy::methods::QueryEventInfo::Output output,
                   ara::core::Vector<rtf::maintaind::EventInfoWithPubSub> &eventInfoWithPubSubListTmp) const;
rtf::maintaind::LatencyIndex QueryIndex(const ara::core::String& eventName, bool& isFindEvent, bool& isOnline);
ara::core::Vector<rtf::maintaind::EventInfoWithPubSub> QueryEventInfo(const ara::core::String& eventName);
void FilterByInstanceName(
    ara::core::Vector<rtf::maintaind::EventInfoWithPubSub>& eventInfoWithPubSubList,
    const ara::core::String& pubName) const;
const std::uint32_t waitMethodRequest_ = 300;
bool isInit_ = false;
bool hasIndex_ = false;
rtf::maintaind::LatencyIndex latencyIndex_;
std::mutex initMutex_;
std::shared_ptr<rtf::rtftools::common::ToolsCommonClientManager> toolsCommonClientManager_ = nullptr;
};
}
}

#endif
