/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2022. All rights reserved.
 * Module: ssm
 * Created on: Oct 10, 2023
 * Author: aviroz
 */

#ifndef _DISPATCHER_H
#define _DISPATCHER_H

#include "sys_statemgr/include/ppscontrol.h"
#include "sys_statemgr/include/DsvPpsIfStruct.h"
#include "sys_statemgr/include/sys_define.h"
#include "state_manager.h"
#include <stdio.h>
#include <signal.h>

using namespace Dsvpps;
using namespace DESY;

namespace hozon {
namespace netaos {
namespace ssm {

class Dispatcher : public HalSubInterface
{

public:
    Dispatcher();
    virtual ~Dispatcher();
    void Init(std::shared_ptr<StateManager>);
    void InitPPS();
    void DeInit();
    void Run();

    void SendPowerMgrMsg2DSVData(const DsvPowerMgrMsg_Array *arry);

private:
    void onHalSubInterface(int topicid, int cmdid, int size, char *payload) override;
    static void stateCallback(int topicid, int param, int errid, char *errstr);
    void PowerMsgHandle(DsvPowerMgrMsg_Array *arry);
    bool DecodeDsvPowerMgrMsg(DsvPowerMgrMsg_Array *arry);
    void EncodeDsvPowerMgrMsg(DsvPowerMgrMsg_Array *arry);

    void EmptyQueueSocModeState();
    void EnqueueSocModeState(SocModeState);
    SocModeState DequeueSocModeState();

    uint16_t CRC16(uint8_t *data, uint16_t len);

public:
    std::shared_ptr<StateManager> m_smgr;
    ppscontrol * m_ppsctl;
    DsvPowerMgrMsg_Array m_arry;
    std::queue<SocModeState> m_que_mstate;
    std::mutex m_mutex_mstate;
    std::thread m_thr_pm;
    sig_atomic_t m_stopFlag;
};

}}}
#endif // _DISPATCHER_H