/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: sm
 * Description: state_client_impl_zmq.h
 * Created on: Nov 16, 2023
 * Author: yanlongxiang
 * 
 */
#ifndef STATE_CLIENT_IMPL_ZMQ_H
#define STATE_CLIENT_IMPL_ZMQ_H
#include <map>
#include "sm/include/sm_types.h"
#include "sm/include/sm_logger.h"
#include "em/include/proctypes.h"
#include "zmq_ipc/manager/zmq_ipc_client.h"
#include "zmq_ipc/proto/sm.pb.h"

using namespace hozon::netaos::em;
using namespace hozon::netaos::log;

namespace hozon {
namespace netaos{
namespace sm{

using PreProcessFunc = std::function<int32_t(const std::string& old_mode, const std::string& new_mode)>;
using PostProcessFunc = std::function<void(const std::string& old_mode, const std::string& new_mode, const bool succ)>;

const std::string stat_client_name = "tcp://localhost:11150";
class StateClientImplZmq {
public:

    StateClientImplZmq();
    ~StateClientImplZmq();
    int32_t RegisterPreProcessFunc(const std::string& old_mode, const std::string& new_mode, PreProcessFunc f);
    int32_t RegisterPostProcessFunc(const std::string& old_mode, const std::string& new_mode, PostProcessFunc f);
    int32_t SwitchMode(const std::string& new_mode);
    int32_t StopMode();
    int32_t GetCurrMode(std::string& curr_mode);
    int32_t SetDefaultMode(const std::string& default_mode);
    int32_t GetProcessInfo(vector<ProcessInfo> &process_info);
    vector<std::string> GetAdfLiteProcess();
    int32_t ProcRestart(const std::string& proc_name);
    int32_t GetModeList(std::vector<std::string> &mode_list);
    int32_t GetModeListDetailInfo(std::vector<hozon::netaos::zmqipc::process_info> &process_info_list);
    std::string GetProcessName();

private:
    std::string processname_fordebug;
    int32_t SendRequestToServer(const uint32_t& type, const uint32_t& reply_type, const std::string& old_mode="", const std::string& new_mode="", const bool& succ=true, const string& extra_data="");
private:
    map<string,PreProcessFunc> PreProcessFuncMap;
    map<string,PostProcessFunc> PostProcessFuncMap;

    std::shared_ptr<hozon::netaos::zmqipc::ZmqIpcClient> client_zmq_;
    hozon::netaos::zmqipc::sm_reply resp_data_{};
};
} // namespace sm
} // namespace netaos
} // namespace hozon
#endif
