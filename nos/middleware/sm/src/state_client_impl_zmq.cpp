/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: sm
 * Description: state_client_impl.cpp
 * Created on: Feb 7, 2023
 * Author: renhongyan
 *
 */
#include <iostream>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "sm/include/sm_types.h"
#include "sm/include/state_client_impl_zmq.h"

extern char **environ;

using namespace std;
using namespace hozon::netaos::log;
using namespace hozon::netaos::em;
using namespace hozon::netaos::sm;
namespace hozon {
namespace netaos{
namespace sm{

StateClientImplZmq::StateClientImplZmq()
{
    std::string topic_name = GetProcessName() + TOPIC_SUFFIX_NAME;
    SM_CLIENT_LOG_INFO << "preprocess postprocess callback server start, topic is [" + topic_name + "], pid is [" << getpid() << "]";

    client_zmq_ = std::make_shared<hozon::netaos::zmqipc::ZmqIpcClient>();
    client_zmq_->Init(stat_client_name);
}

StateClientImplZmq::~StateClientImplZmq() {
    SM_CLIENT_LOG_INFO <<  "StateClientImplZmq.Deinit()";
    client_zmq_->Deinit();
    SM_CLIENT_LOG_INFO <<  "StateClientImplZmq.Deinit() over";
}

int32_t StateClientImplZmq::SendRequestToServer(const uint32_t& type, const uint32_t& reply_type, const std::string& old_mode, const std::string& new_mode, const bool& succ, const string& extra_data)
{
    SM_CLIENT_LOG_DEBUG << "SendRequestToServerZmq: type = " << FormatType(type) << ", old_mode is [" << old_mode << "], new_mode is [" << new_mode << "],extra_data is [" << extra_data << "]";

    hozon::netaos::zmqipc::sm_request req_data{};
    req_data.set_type(type);
    req_data.set_process_name(GetProcessName());
    req_data.set_old_mode(old_mode);
    req_data.set_new_mode(new_mode);
    req_data.set_succ(succ);
    req_data.set_extra_data(extra_data);
    std::string reply{};
    int32_t res = client_zmq_->Request(req_data.SerializeAsString(), reply, 20000);
    if (res == 0) {
        resp_data_.ParseFromString(reply);
        SM_CLIENT_LOG_DEBUG << "SendRequestToServerZmq: client_zmq_.Request result type is " << FormatType(resp_data_.type()) << ",;process name is [" << resp_data_.process_name() << "]";
        SM_CLIENT_LOG_DEBUG << "SendRequestToServerZmq: client_zmq_.Request result is OK, res is " << res;
        if (resp_data_.type() == reply_type) {
            if (resp_data_.result() == 0) {
                SM_CLIENT_LOG_DEBUG << "Request " << FormatType(type) << ", reply type " << FormatType(resp_data_.type()) << ", is OK, and result[" << resp_data_.result() << "] is success";
                res = static_cast<int32_t>(SmResultCode::kSuccess);
            } else {
                SM_CLIENT_LOG_ERROR << "Request " << FormatType(type) << ", reply type " << FormatType(resp_data_.type()) << ", is OK, but result[" << resp_data_.result() << "] is fail";
                res = resp_data_.result();
            }
        } else {
            SM_CLIENT_LOG_ERROR << "Request " << FormatType(type) << ", reply type " << FormatType(resp_data_.type()) << ", is NG, result is [" << resp_data_.result() << "]";
            res = static_cast<int32_t>(SmResultCode::kInvalid);
        }
    } else {
        SM_CLIENT_LOG_ERROR << "SendRequestToServerZmq: client_zmq_.Request " << FormatType(type) << ", result [" << res << "] is NG";
        res = static_cast<int32_t>(SmResultCode::kTimeout);
    }
    return res;
}
int32_t StateClientImplZmq::RegisterPreProcessFunc(const std::string& old_mode, const std::string& new_mode, PreProcessFunc f)
{
    SM_CLIENT_LOG_INFO << "No.1 StateClientImplZmq::RegisterPreProcessFunc is called";
    std::string preprocess_func_name = GetProcessName() + "_" + old_mode + "_" + new_mode + "__preprocess";
    PreProcessFuncMap.insert(make_pair(preprocess_func_name, f));
    SM_CLIENT_LOG_INFO << "StateClientImplZmq Registered preprocess func in local [" + preprocess_func_name << "] for (" << old_mode << "," << new_mode << "] success";
    int32_t res = SendRequestToServer(REQUEST_CODE_REGISTER_PREPROCESS_FUNC, REPLY_CODE_REGISTER_PREPROCESS_FUNC, old_mode, new_mode, true, preprocess_func_name);
    SM_CLIENT_LOG_INFO << "StateClientImplZmq Registered preprocess func in server [" + preprocess_func_name << "] for (" << old_mode << "," << new_mode << ") over, result is [" << res << "]";
    return res;
}

int32_t StateClientImplZmq::RegisterPostProcessFunc(const std::string& old_mode, const std::string& new_mode, PostProcessFunc f)
{
    SM_CLIENT_LOG_INFO << "No.1 StateClientImplZmq::RegisterPostProcessFunc is called";
    std::string postprocess_func_name = GetProcessName() + "_" + old_mode + "_" + new_mode + "__postprocess";
    PostProcessFuncMap.insert(make_pair(postprocess_func_name, f));
    // SM_CLIENT_LOG_INFO << "StateClientImplZmq Registered postprocess func in local [" + postprocess_func_name << "] for (" << old_mode << "," << new_mode << "] success";
    int32_t res = SendRequestToServer(REQUEST_CODE_REGISTER_POSTPROCESS_FUNC, REPLY_CODE_REGISTER_POSTPROCESS_FUNC, old_mode, new_mode, true, postprocess_func_name);
    SM_CLIENT_LOG_INFO << "StateClientImplZmq Registered postprocess func in server [" + postprocess_func_name << "] for (" << old_mode << "," << new_mode << ") over, result is [" << res << "]";
    return res;
}

int32_t StateClientImplZmq::SwitchMode(const std::string& new_mode)
{
    SM_CLIENT_LOG_INFO << "No.1 StateClientImplZmq::SwitchMode is called";
    int32_t res = SendRequestToServer(REQUEST_CODE_SWITCH_MODE, REPLY_CODE_SWITCH_MODE, "", new_mode);
    SM_CLIENT_LOG_INFO << "StateClientImplZmq SwitchMode to [" + new_mode << "] over, result is [" << res << "]";
    return res;
}

int32_t StateClientImplZmq::StopMode()
{
    SM_CLIENT_LOG_INFO << "No.1 StateClientImplZmq::StopMode is called";
    int32_t res = SendRequestToServer(REQUEST_CODE_STOP_MODE, REPLY_CODE_STOP_MODE);
    SM_CLIENT_LOG_INFO << "StateClientImplZmq StopMode over, result is [" << res << "]";
    return res;
}

int32_t StateClientImplZmq::GetCurrMode(std::string& curr_mode)
{
    SM_CLIENT_LOG_DEBUG << "No.1 StateClientImplZmq::GetCurrMode is called";
    int32_t res = SendRequestToServer(REQUEST_CODE_GET_CURR_MODE, REPLY_CODE_GET_CURR_MODE);
    SM_CLIENT_LOG_DEBUG << "StateClientImplZmq GetCurrMode is [" + curr_mode << "] over, result is [" << res << "]";
    if (res == 0) {
        curr_mode = resp_data_.curr_mode();
    }
    return res;
}

int32_t StateClientImplZmq::SetDefaultMode(const std::string& default_mode)
{
    SM_CLIENT_LOG_INFO << "No.1 StateClientImplZmq::SetDefaultMode is called";
    int32_t res = SendRequestToServer(REQUEST_CODE_SET_DEFAULT_MODE, REPLY_CODE_SET_DEFAULT_MODE, "", default_mode);
    SM_CLIENT_LOG_INFO << "StateClientImplZmq SetDefaultMode to [" + default_mode << "] over, result is [" << res << "]";
    return res;
}

int32_t StateClientImplZmq::GetProcessInfo(vector<ProcessInfo> &process_info)
{
    SM_CLIENT_LOG_DEBUG << "StateClientImplZmq::GetProcessInfo is called";
    int32_t res = SendRequestToServer(REQUEST_CODE_GET_PROC_INFO, REPLY_CODE_GET_PROC_INFO);
    if (res == 0) {
        SM_CLIENT_LOG_DEBUG << "StateClientImplZmq::GetProcessInfo success";
        process_info.clear();
        for (auto proc_info: resp_data_.data()) {
            //SM_CLIENT_LOG_INFO << "Copy ProcessInfo";
            ProcessInfo proc_info2 {proc_info.group(), proc_info.procname(), static_cast<ProcessState>(proc_info.procstate())};
            process_info.emplace_back(proc_info2);
        }
    }
    //SM_CLIENT_LOG_DEBUG << "StateClientImplZmq::GetProcessInfo is over";
    return res;
}

vector<std::string> StateClientImplZmq::GetAdfLiteProcess()
{
    vector<std::string> adf_lite_process;
    vector<ProcessInfo> process_infos;
    int32_t res = GetProcessInfo(process_infos);
    if (res == 0) {
        for (auto proc_info: process_infos) {
            if (proc_info.procstate == ProcessState::RUNNING) {
                adf_lite_process.emplace_back(proc_info.procname);
            }
        }
    } else {
        SM_CLIENT_LOG_ERROR << "GetProcessInfo has error!";
    }
    return adf_lite_process;
}

int32_t StateClientImplZmq::ProcRestart(const std::string& proc_name)
{
    SM_CLIENT_LOG_INFO << "StateClientImplZmq::ProcRestart to [" << proc_name << "] is called";
    int32_t res = SendRequestToServer(REQUEST_CODE_PROC_RESTART, REPLY_CODE_PROC_RESTART, "", "", true, proc_name);
    SM_CLIENT_LOG_INFO << "StateClientImplZmq ProcRestart to [" << proc_name << "] over, result is [" << res << "]";
    return res;
}

int32_t StateClientImplZmq::GetModeList(std::vector<std::string> &mode_list)
{
    SM_CLIENT_LOG_INFO << "StateClientImplZmq::GetModeList is called";
    int32_t res = SendRequestToServer(REQUEST_CODE_GET_MODE_LIST, REPLY_CODE_GET_MODE_LIST);
    if (res == 0) {
        SM_CLIENT_LOG_INFO << "StateClientImplZmq::GetModeList success";
        mode_list.clear();
        for (auto mode_info: resp_data_.mode_list()) {
            mode_list.emplace_back(mode_info);
        }
    }
    //SM_CLIENT_LOG_INFO << "StateClientImplZmq::GetModeList is over";
    return res;
}

int32_t StateClientImplZmq::GetModeListDetailInfo(std::vector<hozon::netaos::zmqipc::process_info> &process_info_list) {
    SM_CLIENT_LOG_INFO << "StateClientImplZmq::GetModeListDetailInfo is called";
    int32_t res = SendRequestToServer(REQUEST_CODE_GET_MODE_LIST_DETAIL_INFO, REQUEST_CODE_GET_MODE_LIST_DETAIL_INFO);
    if (res == 0) {
        SM_CLIENT_LOG_INFO << "StateClientImplZmq::GetModeListDetailInfo success";
        for (auto i = 0; i < resp_data_.data_size(); ++i) {
            process_info_list.emplace_back(resp_data_.data(i));
        }
    }
    return res;
}

std::string StateClientImplZmq::GetProcessName()
{
    if (processname_fordebug != "") {
        return processname_fordebug;
    }
    std::string pname = environ[0];
    size_t pos = pname.find(ENVRION_NAME);
    if(pos != pname.npos){
        pname.replace(pos,strlen(ENVRION_NAME),"");
    }else{
        SM_CLIENT_LOG_DEBUG <<"< get environ proc name fail! >";
    }
    return pname;
}

} // namespace sm
} // namespace netaos
} // namespace hozon
