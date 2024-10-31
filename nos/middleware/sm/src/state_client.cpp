/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: sm
 * Description: state_client.cpp
 * Created on: Jun 7, 2022
 * Author: renhongyan
 * 
 */
#include "sm/include/state_client.h"
#include "sm/include/state_client_impl.h"

using namespace std;
namespace hozon {
namespace netaos{
namespace sm{

StateClient::StateClient()
{
    _pimpl = std::make_unique<StateClientImpl>();
}

StateClient::~StateClient() {
}

int32_t StateClient::RegisterPreProcessFunc(const std::string& old_mode, const std::string& new_mode, PreProcessFunc f)
{
    return _pimpl->RegisterPreProcessFunc(old_mode, new_mode, f);
}

int32_t StateClient::RegisterPostProcessFunc(const std::string& old_mode, const std::string& new_mode, PostProcessFunc f)
{
    return _pimpl->RegisterPostProcessFunc(old_mode, new_mode, f);
}

int32_t StateClient::SwitchMode(const std::string& new_mode)
{
    return _pimpl->SwitchMode(new_mode);
}

int32_t StateClient::StopMode()
{
    return _pimpl->StopMode();
}

int32_t StateClient::GetCurrMode(std::string& curr_mode)
{
    return _pimpl->GetCurrMode(curr_mode);
}

int32_t StateClient::SetDefaultMode(const std::string& default_mode)
{
    return _pimpl->SetDefaultMode(default_mode);
}

int32_t StateClient::GetProcessInfo(vector<ProcessInfo> &process_info)
{
    return _pimpl->GetProcessInfo(process_info);
}

vector<std::string> StateClient::GetAdfLiteProcess()
{
    return _pimpl->GetAdfLiteProcess();
}

int32_t StateClient::ProcRestart(const std::string &proc_name)
{
    return _pimpl->ProcRestart(proc_name);
}

int32_t StateClient::GetModeList(vector<string> &mode_list)
{
    return _pimpl->GetModeList(mode_list);
}

void StateClient::SetProcessName(const std::string &proc_name)
{
    return _pimpl->SetProcessName(proc_name);
}

std::string StateClient::GetProcessName()
{
    return _pimpl->GetProcessName();
}
} // namespace sm
} // namespace netaos
} // namespace hozon
