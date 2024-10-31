/*
* Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
* Description: SM API only for MDC platform state
* Create: 2022-04-11
* Notes: NA
*/
#ifndef ARA_SM_PLATFORM_STATE_CLIENT_H
#define ARA_SM_PLATFORM_STATE_CLIENT_H

#include <memory>

#include "ara/sm/sm_common.h"

namespace ara {
namespace sm {
class PlatformStateClient final {
public:
    PlatformStateClient();
    ~PlatformStateClient();
    PlatformStateClient(const PlatformStateClient&) = delete;
    PlatformStateClient& operator=(const PlatformStateClient&) = delete;
    PlatformStateClient(PlatformStateClient&&) = default;
    PlatformStateClient& operator=(PlatformStateClient&&) = default;

    SmResultCode Init();

    SmResultCode RequestPlatformState(const ara::sm::PlatformState& platformStateIn,
        const ara::core::Vector<uint32_t>& data = {});

    SmResultCode InquirePlatformState(ara::sm::PlatformState& platformStateOut);

    SmResultCode RegisterPlatformStateNotifyHandler(
        const std::function<void (ara::sm::PlatformState platformStateIn,
                                  ara::sm::SmResultCode returnType)>& handler);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};
}
}
#endif