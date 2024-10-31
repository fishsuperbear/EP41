/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: sm
 * Description: state_client.h
 * Created on: Feb 7, 2023
 * Author: renhongyan
 * 
 */
#ifndef STATE_CLIENT_H
#define STATE_CLIENT_H

#include <functional>
#include <memory>
#include <vector>
#include "em/include/proctypes.h"
#include "state_client_impl.h"

namespace hozon {
namespace netaos{
namespace sm{

using namespace hozon::netaos::em;
using PreProcessFunc = std::function<int32_t(const std::string& old_mode, const std::string& new_mode)>;
using PostProcessFunc = std::function<void(const std::string& old_mode, const std::string& new_mode, const bool succ)>;

class StateClientImpl;
class StateClient {
public:

    StateClient();
    ~StateClient();
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
    void SetProcessName(const std::string &proc_name);
    std::string GetProcessName();

private:
    std::unique_ptr<StateClientImpl> _pimpl;

};
} // namespace sm
} // namespace netaos
} // namespace hozon
#endif