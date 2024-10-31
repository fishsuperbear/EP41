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
#include "sm/include/state_client_impl.h"

extern char **environ;

using namespace std;
using namespace hozon::netaos::log;
using namespace hozon::netaos::em;
using namespace hozon::netaos::sm;
namespace hozon {
namespace netaos{
namespace sm{

int32_t StateClientImpl::RequestProcess(const std::shared_ptr<sm_request> req, std::shared_ptr<sm_reply> resp)
{
    if (req->type() == REQUEST_CODE_PREPROCESS_FUNC) {
        SM_CLIENT_LOG_INFO << "received RegisterPreProcessFunc callback: type = " << FormatType(req->type()) << ", func_name is [" << req->extra_data() << "]";
        if (PreProcessFuncMap.count(req->extra_data()) == 0) {
            SM_CLIENT_LOG_INFO <<  "Current process [" + GetProcessName() + "] has not register preprocess func_name:" + req->extra_data();
            return 0;
        }
        PreProcessFunc f_pre = PreProcessFuncMap[req->extra_data()];
        int32_t res = f_pre(req->old_mode(), req->new_mode());
        SM_CLIENT_LOG_INFO << "client preprocess callback function over, res is [" << res << "]";
        resp->type(REPLY_CODE_PREPROCESS_FUNC);
        resp->process_name(GetProcessName());
        resp->result(res);
        SM_CLIENT_LOG_INFO << "resp->type is " << FormatType(resp->type()) << ", resp->process_name() is [" << resp->process_name() << "], res is [" << res << "]";
    } else if (req->type() == REQUEST_CODE_POSTPROCESS_FUNC) {
        SM_CLIENT_LOG_INFO << "received RegisterPostProcessFunc callback: type = " << FormatType(req->type()) << ", func_name is [" << req->extra_data() << "]";
        if (PostProcessFuncMap.count(req->extra_data()) == 0) {
            SM_CLIENT_LOG_INFO <<  "Current process [" + GetProcessName() + "] has not register postprocess func_name:" + req->extra_data();
            return 0;
        }
        PostProcessFunc f_post = PostProcessFuncMap[req->extra_data()];
        f_post(req->old_mode(), req->new_mode(), req->succ());
        SM_CLIENT_LOG_INFO << "client postprocess callback function over";
        resp->type(REPLY_CODE_POSTPROCESS_FUNC);
        resp->process_name(GetProcessName());
        SM_CLIENT_LOG_INFO << "resp->type is " << FormatType(resp->type()) << ", resp->process_name() is [" << resp->process_name() << "]";
    } else {
        SM_CLIENT_LOG_ERROR << "received RequestProcess callback: type [" << req->type() << "] is unhandled";
    }

    SM_CLIENT_LOG_INFO <<  "RequestProcess::over";
    return 0;
}

StateClientImpl::StateClientImpl():
        req_data_type(std::make_shared<sm_requestPubSubType>()),
        resp_data_type(std::make_shared<sm_replyPubSubType>()),
        req_data(std::make_shared<sm_request>()),
        resp_data(std::make_shared<sm_reply>()),
        client(std::make_shared<sm_requestPubSubType>(), std::make_shared<sm_replyPubSubType>()),
        process_func_server(std::make_shared<sm_requestPubSubType>(), std::make_shared<sm_replyPubSubType>(), *this)
{
    std::string topic_name = GetProcessName() + TOPIC_SUFFIX_NAME;
    SM_CLIENT_LOG_INFO << "preprocess postprocess callback server start, topic is [" + topic_name + "], pid is [" << getpid() << "]";
    process_func_server.Start(0, topic_name);

    client.Init(0, "state_client_request");
    client.WaitServiceOnline(5000);
}

StateClientImpl::~StateClientImpl() {
    SM_CLIENT_LOG_INFO <<  "StateClientImpl.Deinit()";
    process_func_server.Stop();
    client.Deinit();
    SM_CLIENT_LOG_INFO <<  "StateClientImpl.Deinit() over";
}

int32_t StateClientImpl::SendRequestToServer(const uint32_t& type, const uint32_t& reply_type, const std::string& old_mode, const std::string& new_mode, const bool& succ, const string& extra_data)
{
    SM_CLIENT_LOG_DEBUG << "SendRequestToServer: type = " << FormatType(type) << ", old_mode is [" << old_mode << "], new_mode is [" << new_mode << "],extra_data is [" << extra_data << "]";

    client.WaitServiceOnline(1000);
    req_data->fire_forget(false);
    req_data->type(type);
    req_data->process_name(GetProcessName());
    req_data->old_mode(old_mode);
    req_data->new_mode(new_mode);
    req_data->succ(succ);
    req_data->extra_data(extra_data);
    int32_t res = client.Request(req_data, resp_data, 5000 * 5);
    if (res == 0) {
        SM_CLIENT_LOG_DEBUG << "SendRequestToServer: client.Request result type is " << FormatType(resp_data->type()) << ",;process name is [" << resp_data->process_name() << "]";
        SM_CLIENT_LOG_DEBUG << "SendRequestToServer: client.Request result is OK, res is " << res;
        if (resp_data->type() == reply_type) {
            if (resp_data->result() == 0) {
                SM_CLIENT_LOG_DEBUG << "Request " << FormatType(type) << ", reply type " << FormatType(resp_data->type()) << ", is OK, and result[" << resp_data->result() << "] is success";
                res = static_cast<int32_t>(SmResultCode::kSuccess);
            } else {
                SM_CLIENT_LOG_ERROR << "Request " << FormatType(type) << ", reply type " << FormatType(resp_data->type()) << ", is OK, but result[" << resp_data->result() << "] is fail";
                res = resp_data->result();
            }
        } else {
            SM_CLIENT_LOG_ERROR << "Request " << FormatType(type) << ", reply type " << FormatType(resp_data->type()) << ", is NG, result is [" << resp_data->result() << "]";
            res = static_cast<int32_t>(SmResultCode::kInvalid);
        }
    } else {
        SM_CLIENT_LOG_ERROR << "SendRequestToServer: client.Request " << FormatType(type) << ", result [" << res << "] is NG";
        res = static_cast<int32_t>(SmResultCode::kTimeout);
    }
    return res;
}

int32_t StateClientImpl::RegisterPreProcessFunc(const std::string& old_mode, const std::string& new_mode, PreProcessFunc f)
{
    SM_CLIENT_LOG_INFO << "No.1 StateClientImpl::RegisterPreProcessFunc is called";
    std::string preprocess_func_name = GetProcessName() + "_" + old_mode + "_" + new_mode + "__preprocess";
    PreProcessFuncMap.insert(make_pair(preprocess_func_name, f));
    SM_CLIENT_LOG_INFO << "StateClientImpl Registered preprocess func in local [" + preprocess_func_name << "] for (" << old_mode << "," << new_mode << "] success";
    int32_t res = SendRequestToServer(REQUEST_CODE_REGISTER_PREPROCESS_FUNC, REPLY_CODE_REGISTER_PREPROCESS_FUNC, old_mode, new_mode, true, preprocess_func_name);
    SM_CLIENT_LOG_INFO << "StateClientImpl Registered preprocess func in server [" + preprocess_func_name << "] for (" << old_mode << "," << new_mode << ") over, result is [" << res << "]";
    return res;
}

int32_t StateClientImpl::RegisterPostProcessFunc(const std::string& old_mode, const std::string& new_mode, PostProcessFunc f)
{
    SM_CLIENT_LOG_INFO << "No.1 StateClientImpl::RegisterPostProcessFunc is called";
    std::string postprocess_func_name = GetProcessName() + "_" + old_mode + "_" + new_mode + "__postprocess";
    PostProcessFuncMap.insert(make_pair(postprocess_func_name, f));
    // SM_CLIENT_LOG_INFO << "StateClientImpl Registered postprocess func in local [" + postprocess_func_name << "] for (" << old_mode << "," << new_mode << "] success";
    int32_t res = SendRequestToServer(REQUEST_CODE_REGISTER_POSTPROCESS_FUNC, REPLY_CODE_REGISTER_POSTPROCESS_FUNC, old_mode, new_mode, true, postprocess_func_name);
    SM_CLIENT_LOG_INFO << "StateClientImpl Registered postprocess func in server [" + postprocess_func_name << "] for (" << old_mode << "," << new_mode << ") over, result is [" << res << "]";
    return res;
}

int32_t StateClientImpl::SwitchMode(const std::string& new_mode)
{
    SM_CLIENT_LOG_INFO << "No.1 StateClientImpl::SwitchMode is called";
    int32_t res = SendRequestToServer(REQUEST_CODE_SWITCH_MODE, REPLY_CODE_SWITCH_MODE, "", new_mode);
    SM_CLIENT_LOG_INFO << "StateClientImpl SwitchMode to [" + new_mode << "] over, result is [" << res << "]";
    return res;
}

int32_t StateClientImpl::StopMode()
{
    SM_CLIENT_LOG_INFO << "No.1 StateClientImpl::StopMode is called";
    int32_t res = SendRequestToServer(REQUEST_CODE_STOP_MODE, REPLY_CODE_STOP_MODE);
    SM_CLIENT_LOG_INFO << "StateClientImpl StopMode over, result is [" << res << "]";
    return res;
}

int32_t StateClientImpl::GetCurrMode(std::string& curr_mode)
{
    SM_CLIENT_LOG_DEBUG << "No.1 StateClientImpl::GetCurrMode is called";
    int32_t res = SendRequestToServer(REQUEST_CODE_GET_CURR_MODE, REPLY_CODE_GET_CURR_MODE);
    SM_CLIENT_LOG_DEBUG << "StateClientImpl GetCurrMode is [" + curr_mode << "] over, result is [" << res << "]";
    if (res == 0) {
        curr_mode = resp_data->curr_mode();
    }
    return res;
}

int32_t StateClientImpl::SetDefaultMode(const std::string& default_mode)
{
    SM_CLIENT_LOG_INFO << "No.1 StateClientImpl::SetDefaultMode is called";
    int32_t res = SendRequestToServer(REQUEST_CODE_SET_DEFAULT_MODE, REPLY_CODE_SET_DEFAULT_MODE, "", default_mode);
    SM_CLIENT_LOG_INFO << "StateClientImpl SetDefaultMode to [" + default_mode << "] over, result is [" << res << "]";
    return res;
}

int32_t StateClientImpl::GetProcessInfo(vector<ProcessInfo> &process_info)
{
    SM_CLIENT_LOG_DEBUG << "StateClientImpl::GetProcessInfo is called";
    int32_t res = SendRequestToServer(REQUEST_CODE_GET_PROC_INFO, REPLY_CODE_GET_PROC_INFO);
    if (res == 0) {
        SM_CLIENT_LOG_DEBUG << "StateClientImpl::GetProcessInfo success";
        process_info.clear();
        for (auto proc_info: resp_data->data()) {
            //SM_CLIENT_LOG_INFO << "Copy ProcessInfo";
            ProcessInfo proc_info2 {proc_info.group(), proc_info.procname(), static_cast<ProcessState>(proc_info.procstate())};
            process_info.emplace_back(proc_info2);
        }
    }
    //SM_CLIENT_LOG_DEBUG << "StateClientImpl::GetProcessInfo is over";
    return res;
}

vector<std::string> StateClientImpl::GetAdfLiteProcess()
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

int32_t StateClientImpl::ProcRestart(const std::string& proc_name)
{
    SM_CLIENT_LOG_INFO << "StateClientImpl::ProcRestart to [" << proc_name << "] is called";
    int32_t res = SendRequestToServer(REQUEST_CODE_PROC_RESTART, REPLY_CODE_PROC_RESTART, "", "", true, proc_name);
    SM_CLIENT_LOG_INFO << "StateClientImpl ProcRestart to [" << proc_name << "] over, result is [" << res << "]";
    return res;
}

int32_t StateClientImpl::GetModeList(std::vector<std::string> &mode_list)
{
    SM_CLIENT_LOG_INFO << "StateClientImpl::GetModeList is called";
    int32_t res = SendRequestToServer(REQUEST_CODE_GET_MODE_LIST, REPLY_CODE_GET_MODE_LIST);
    if (res == 0) {
        SM_CLIENT_LOG_INFO << "StateClientImpl::GetModeList success";
        mode_list.clear();
        for (auto mode_info: resp_data->mode_list()) {
            mode_list.emplace_back(mode_info);
        }
    }
    //SM_CLIENT_LOG_INFO << "StateClientImpl::GetModeList is over";
    return res;
}

std::string StateClientImpl::GetProcessName()
{
#ifdef UT
    return "StateClientImpl";
#endif

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
