/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: sm
 * Description: state_client.cpp
 * Created on: Jun 7, 2022
 * Author: renhongyan
 * 
 */
#include "sm/include/state_client_zmq.h"
#include "sm/include/state_client_impl_zmq.h"

using namespace std;
namespace hozon {
namespace netaos{
namespace sm{

StateClientZmq::StateClientZmq()
{
    _pimpl = std::make_unique<StateClientImplZmq>();
}

StateClientZmq::~StateClientZmq() {
}

int32_t StateClientZmq::RegisterPreProcessFunc(const std::string& old_mode, const std::string& new_mode, PreProcessFunc f)
{
    return _pimpl->RegisterPreProcessFunc(old_mode, new_mode, f);
}

int32_t StateClientZmq::RegisterPostProcessFunc(const std::string& old_mode, const std::string& new_mode, PostProcessFunc f)
{
    return _pimpl->RegisterPostProcessFunc(old_mode, new_mode, f);
}

int32_t StateClientZmq::SwitchMode(const std::string& new_mode)
{
    return _pimpl->SwitchMode(new_mode);
}

int32_t StateClientZmq::StopMode()
{
    return _pimpl->StopMode();
}

int32_t StateClientZmq::GetCurrMode(std::string& curr_mode)
{
    return _pimpl->GetCurrMode(curr_mode);
}

int32_t StateClientZmq::SetDefaultMode(const std::string& default_mode)
{
    return _pimpl->SetDefaultMode(default_mode);
}

int32_t StateClientZmq::GetProcessInfo(vector<ProcessInfo> &process_info)
{
    return _pimpl->GetProcessInfo(process_info);
}

vector<std::string> StateClientZmq::GetAdfLiteProcess()
{
    return _pimpl->GetAdfLiteProcess();
}

int32_t StateClientZmq::ProcRestart(const std::string &proc_name)
{
    return _pimpl->ProcRestart(proc_name);
}

int32_t StateClientZmq::GetModeList(vector<string> &mode_list)
{
    return _pimpl->GetModeList(mode_list);
}

int32_t StateClientZmq::GetModeListDetailInfo(std::vector<hozon::netaos::zmqipc::process_info> &process_info_list) {
    return _pimpl->GetModeListDetailInfo(process_info_list);
}

std::string StateClientZmq::GetProcessName()
{
    return _pimpl->GetProcessName();
}
} // namespace sm
} // namespace netaos
} // namespace hozon
