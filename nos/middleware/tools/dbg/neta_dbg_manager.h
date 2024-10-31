/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2022. All rights reserved.
 * Module: dbg
 * Created on: May 8, 2023
 */

#ifndef NETA_DBG_MANAGER_H
#define NETA_DBG_MANAGER_H

#include <memory>
#include "conf.h"
#include "log/include/logging.h"
#include "em/include/proctypes.h"
#include "sm/include/state_client_zmq.h"

namespace hozon {
namespace netaos {
namespace dbg {

using namespace hozon::netaos::sm;
using namespace hozon::netaos::em;
using namespace hozon::netaos::log;

class NetaDbgManager {
public:

    NetaDbgManager();
    virtual ~NetaDbgManager();

    int32_t Init();
    void DeInit();

    void Help();
    void RestartProcess(std::string &);
    void SwitchMode(std::string &);
    void StopMode();
    void GetModeList();
    void GetModeListDetailInfo();
    void GetCurrentMode();
    void GetProcessState();
    void SetDefaultStartupMode(std::string &);

    int32_t Reboot();
    int32_t Reset();

    void Sysdate();

private:
    static bool Sortcmp(ProcessInfo &, ProcessInfo &);
    static uint32_t CombResult(size_t n,size_t m);
    static uint32_t CombChars(size_t n,size_t m);
    static uint32_t SumString(size_t len,string str);

private:
    std::shared_ptr<StateClientZmq> m_st_cli;
};

}}}

#endif
