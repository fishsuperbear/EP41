/*
* Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
* Description: SM API
* Create: 2020-05-05
* Notes: NA
*/
#ifndef ARA_SM_STATE_CLIENT_H
#define ARA_SM_STATE_CLIENT_H

#include <memory>

#include "ara/sm/sm_common.h"

namespace ara {
namespace sm {
class StateClient final {
public:
    StateClient();
    ~StateClient();

    /* 初始化StateClient服务, 返回结果标识其可用性 */
    SmResultCode Init();

    /* 获取所有功能组名及其状态名 */
    SmResultCode AcquireFunctionGroupInfo(ara::core::Vector<ara::sm::FunctionGroupStates>& functionGroupsInfo);

    /* 查询全部功能组的当前状态 */
    SmResultCode InquireState(ara::core::Vector<ara::sm::StateChange>& functionGroupCurrStates);

    /* 查询指定功能组的当前状态 */
    SmResultCode InquireState(const ara::core::String targetFgName, ara::core::String& currState);

    /* 请求状态转换, 返回值为执行结果, 应用程序可通过async异步获取执行结果 */
    SmResultCode RequestStates(const ara::core::Vector<ara::sm::StateChange> stateChangeList);

    /* 请求状态转换, 调用后立即返回，返回值仅包含服务发现结果，无法获得执行结果 */
    SmResultCode RequestStatesAsync(const ara::core::Vector<ara::sm::StateChange> stateChangeList);

    /* 注册notify处理函数, 作为StateClient类的成员函数, 默认不具备对类外数据的操作权限, notify对象是单一FGS */
    SmResultCode RegisterNotifyHandler(const std::function<void (ara::sm::StateChange stateChangeReq,
        ara::sm::SmResultCode returnType)> handler);

    /* 注册notify处理函数, 允许应用程序只接收自身感兴趣的功能组的状态转换 */
    SmResultCode RegisterNotifyHandler(const std::function<void (ara::sm::StateChange stateChangeReq,
        ara::sm::SmResultCode returnType)> handler,
        const ara::core::Vector<ara::core::String> fgNames);

    /* 查询当前MDC平台是否位于开工状态 */
    SmResultCode IsMdcPlatformReady();

    /* 复位MDC系统 */
    SmResultCode SystemReset(const SysResetCode& sysResetParams,
                               const ara::core::String& user = "mdc",
                               const SysResetCause& sysResetReason = SysResetCause::kNormal);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};
}
}
#endif