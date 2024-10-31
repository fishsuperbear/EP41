/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2022. All rights reserved.
 * Module: ssm
 * Created on: Aug 15, 2023
 * Author: aviroz
 */

#ifndef TEGRA_HV_SYSCTL_H
#define TEGRA_HV_SYSCTL_H

#include <signal.h>
#include <queue>
#include <mutex>
#include <memory>
#include <thread>
#include <shared_mutex>
#include "sys_statemgr/include/tegra_hv_sysmgr.h"

namespace hozon {
namespace netaos {
namespace ssm {

class TegraHvSysCtl {
public:

    TegraHvSysCtl();
    virtual ~TegraHvSysCtl();

    int32_t Init();
    void DeInit();
    void Run();
private:
    void HVSysmsgPreProces(struct hv_sysmgr_message & msg);
    int32_t PMCtrlReboot(bool is_shutdown);
    int32_t PMCtrlSuspend(void);
    int32_t PMCtrlResume(void);
    int32_t PMCtrlRecvMsg(struct hv_sysmgr_message *msg);
    int32_t PMCtrlSendMsg(struct hv_sysmgr_message *msg);
    static void HVPowerModeMonitor(void *arg);
    static void SysMsgHandle(void *arg);
    static void MCUMsgHandle(void *arg);
    static void SSMHeartbeat(void *arg);

    bool IsQueueEmpty();
    void EmptyQueueSSMsg();
    void EnqueueSSMsg(struct hv_sysmgr_message &);
    void DequeueSSMsg(struct hv_sysmgr_message *);

private:
    int m_fd;
    sig_atomic_t m_stopFlag;
    std::mutex m_mutex_sstate;
    std::queue<hv_sysmgr_message> m_que_sstate;
    std::thread m_thr_1;
    std::thread m_thr_2;
    std::thread m_thr_3;
};

}}}
#endif