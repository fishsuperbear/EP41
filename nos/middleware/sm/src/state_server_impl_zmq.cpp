/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: sm
 * Description: state_server_impl.cpp
 * Created on: Feb 7, 2023
 * Author: renhongyan
 *
 */

#include <thread>
#include <chrono>
#include <cstdint>
#include <signal.h>
#include <iostream>
#include <unistd.h>
#include "sm/include/sm_types.h"
#include "em/include/execmanagement.h"
#include "sm/include/state_server_impl_zmq.h"

using namespace hozon::netaos::em;
using namespace hozon::netaos::log;
namespace hozon {
namespace netaos {
namespace sm {


std::string StateServerImplZmq::GetProcessName()
{
    return "StateServerImplZmq";
}

int32_t StateServerImplZmq::Start()
{
    SM_SERVER_LOG_INFO << "StateServerImplZmq::Start";
    state_zmq_server.Start(stat_service_name);
    return 0;
}

void StateServerImplZmq::Stop()
{
    SM_SERVER_LOG_INFO << "StateServerImplZmq::Stop";
    state_zmq_server.Stop();
}

bool StateServerImplZmq::PreAndPostProcess(const ProcessMode mod, const std::string& src_mode, const std::string& tar_mode, std::map<std::string, std::string>& map)
{
    bool ret = true;
    // if (map.size() == 0) {
    //     SM_SERVER_LOG_INFO <<"ignore not need process";
    //     return ret;
    // } else {
    //     int32_t idx = 0;
    //     std::thread thr[map.size()];
    //     std::map<std::string, int32_t> ret_map;

    //     switch (mod) {
    //     case ProcessMode::PREPROCESS: {
    //         SM_SERVER_LOG_INFO << "== PreProcess Start ==";
    //         for (auto it = map.begin(); it != map.end(); idx++, it++) {
    //             SM_SERVER_LOG_INFO << "Request 0x6001 call process [" << it->first << "] registed preprocess func [" << it->second << "]";
    //             const std::string cli_name = it->first;
    //             const std::string extr_data = it->second;
    //             thr[idx] = std::thread ( [this](const std::string& srcmode, const std::string& tarmode, const std::string& cliname, const std::string& extrdata, std::map<std::string, int32_t>* retmap) {
    //                 std::string topic_name = cliname + TOPIC_SUFFIX_NAME;
    //                 client.Init(0, topic_name);
    //                 if (0 != client.WaitServiceOnline(500)) {
    //                     SM_SERVER_LOG_WARN << "svr is not online,topic:"<<topic_name;
    //                 }
    //                 req_data->type(REQUEST_CODE_PREPROCESS_FUNC);
    //                 req_data->process_name(cliname);
    //                 req_data->old_mode(srcmode);
    //                 req_data->new_mode(tarmode);
    //                 req_data->succ(true);
    //                 req_data->extra_data(extrdata);
    //                 req_data->fire_forget(false);
    //                 int res = client.Request(req_data, resp_data, 10000);
    //                 if (res == 0) {
    //                     SM_SERVER_LOG_INFO << "client.Request return value is OK, res is [" << res << "]; resp process name is [" << resp_data->process_name() << "], callback name is " << req_data->extra_data();
    //                     if (resp_data->type() == REPLY_CODE_PREPROCESS_FUNC) {
    //                         if (resp_data->result() == 0) {
    //                             SM_SERVER_LOG_INFO << "Request " << FormatType(REQUEST_CODE_PREPROCESS_FUNC) << ", reply type " << FormatType(resp_data->type()) << ", is OK, and result[" << resp_data->result() << "] is success";
    //                             res = static_cast<int32_t>(SmResultCode::kSuccess);
    //                         } else {
    //                             SM_SERVER_LOG_ERROR << "Request " << FormatType(REQUEST_CODE_PREPROCESS_FUNC) << ", reply type " << FormatType(resp_data->type()) << ", is OK, but result[" << resp_data->result() << "] is fail";
    //                             res = resp_data->result();
    //                         }
    //                     } else {
    //                             SM_SERVER_LOG_INFO << "Request " << FormatType(REQUEST_CODE_PREPROCESS_FUNC) << ", reply type " << FormatType(resp_data->type()) << ", is OK";
    //                             res = static_cast<int32_t>(SmResultCode::kInvalid);
    //                     }
    //                 } else {
    //                     SM_SERVER_LOG_INFO << "client.Request type =" << FormatType(req_data->type()) << "; return value is NG, res is ["
    //                         << res << "]; resp type is " << FormatType(resp_data->type()) << ",; resp process name is [" << resp_data->process_name() << "], callback name is " << req_data->extra_data();
    //                     res = static_cast<int32_t>(SmResultCode::kTimeout);
    //                 }

    //                 retmap->insert({cliname,res});
    //                 client.Deinit();
    //             }, src_mode, tar_mode, cli_name, extr_data, &ret_map);
    //         }

    //         for (size_t i = 0; i < map.size(); i++) {
    //             if (thr[i].joinable()) {
    //                 thr[i].join();
    //             }
    //         }

    //         std::string pre_proc_res = "";
    //         std::map<std::string, int32_t>::iterator itr = ret_map.begin();
    //         for ( ; itr != ret_map.end(); ++itr) {
    //             pre_proc_res += std::string(itr->first) + ":" + std::to_string(itr->second) + " ";
    //             if ( 0 != (int32_t)itr->second ) {
    //                 ret = false;
    //             }
    //         }
    //         SM_SERVER_LOG_INFO << src_mode << "_to_" << tar_mode <<"_preprocess [" << pre_proc_res << "]";
    //         SM_SERVER_LOG_INFO << " == PreProcess Finished ==";
    //     }
    //         break;

    //     case ProcessMode::POSTPROCESS: {
    //         SM_SERVER_LOG_INFO << "== PostProcess Start ==";
    //         for (auto it = map.begin(); it != map.end(); idx++, it++) {
    //             SM_SERVER_LOG_INFO << "Request 0x6002 call process [" << it->first << "] registed postprocess func [" << it->second << "]";
    //             const std::string cli_name = it->first;
    //             const std::string extr_data = it->second;
    //             thr[idx] = std::thread ( [this] ( const std::string& srcmode, const std::string& tarmode, const std::string& cliname, const std::string& extrdata, std::map <std::string, int32_t>* retmap) {
    //                 std::string topic_name = cliname + TOPIC_SUFFIX_NAME;
    //                 client.Init(0, topic_name);
    //                 if (0 != client.WaitServiceOnline(500)) {
    //                     SM_SERVER_LOG_WARN << "svr is not online,topic:"<<topic_name;
    //                 }
    //                 req_data->type(REQUEST_CODE_POSTPROCESS_FUNC);
    //                 req_data->process_name(cliname);
    //                 req_data->old_mode(srcmode);
    //                 req_data->new_mode(tarmode);
    //                 req_data->succ(true);
    //                 req_data->extra_data(extrdata);
    //                 req_data->fire_forget(true);
    //                 int res = client.Request(req_data, resp_data, 10000);
    //                 if (res == 0) {
    //                     SM_SERVER_LOG_INFO << "client.Request return value is OK, res is [" << res << "]; resp process name is [" << resp_data->process_name() << "], callback name is " << req_data->extra_data();
    //                     if (resp_data->type() == REPLY_CODE_POSTPROCESS_FUNC) {
    //                         if (resp_data->result() == 0) {
    //                             SM_SERVER_LOG_INFO << "Request " << FormatType(REQUEST_CODE_POSTPROCESS_FUNC) << ", reply type " << FormatType(resp_data->type()) << ", is OK, and result[" << resp_data->result() << "] is success";
    //                             res = static_cast<int32_t>(SmResultCode::kSuccess);
    //                         } else {
    //                             SM_SERVER_LOG_ERROR << "Request " << FormatType(REQUEST_CODE_POSTPROCESS_FUNC) << ", reply type " << FormatType(resp_data->type()) << ", is OK, but result[" << resp_data->result() << "] is fail";
    //                             res = resp_data->result();
    //                         }
    //                     } else {
    //                             SM_SERVER_LOG_INFO << "Request " << FormatType(REQUEST_CODE_POSTPROCESS_FUNC) << ", reply type " << FormatType(resp_data->type()) << ", is OK";
    //                             res = static_cast<int32_t>(SmResultCode::kInvalid);
    //                     }
    //                 } else {
    //                     SM_SERVER_LOG_INFO << "client.Request type =" << FormatType(req_data->type()) << "; return value is NG, res is ["
    //                         << res << "]; resp type is " << FormatType(resp_data->type()) << ",; resp process name is [" << resp_data->process_name() << "], callback name is " << req_data->extra_data();
    //                     res = static_cast<int32_t>(SmResultCode::kTimeout);
    //                 }

    //                 retmap->insert({cliname,res});
    //                 client.Deinit();
    //             }, src_mode, tar_mode, cli_name, extr_data, &ret_map);
    //         }

    //         for (size_t i = 0; i < map.size(); i++) {
    //             if (thr[i].joinable()) {
    //                 thr[i].join();
    //             }
    //         }

    //         std::string post_proc_res = "";
    //         std::map<std::string, int32_t>::iterator itr = ret_map.begin();
    //         for ( ; itr != ret_map.end(); ++itr) {
    //             post_proc_res += std::string(itr->first) + ":" + std::to_string(itr->second) + " ";
    //             if ( 0 != (int32_t)itr->second ) {
    //                 ret = false;
    //             }
    //         }
    //         SM_SERVER_LOG_INFO << src_mode << "_to_" << tar_mode <<"_postprocess [" << post_proc_res << "]";
    //         SM_SERVER_LOG_INFO << "== PostProcess Finished ==";
    //     }
    //         break;
    //     default:
    //         break;
    //     }
    // }

    return ret;
}

int32_t StateServerImplZmq::RequestProcess(const std::string& request, std::string& reply)
{
    SM_SERVER_LOG_DEBUG << "StateServerImplZmq::RequestProcess received";
    hozon::netaos::zmqipc::sm_request req{};
    req.ParseFromString(request);


    switch(req.type()) {
        case REQUEST_CODE_REGISTER_PREPROCESS_FUNC:
        {
            SM_SERVER_LOG_INFO << "request process name is " << req.process_name() << "; old mode is " << req.old_mode() << "; new_mode is " << req.new_mode();
            {
                std::lock_guard<std::mutex> lock(preprocess_funcmap_mutex);
                auto &it = PreProcessFuncMap[make_pair(req.old_mode(), req.new_mode())];
                it.emplace(req.process_name(), req.extra_data());
                SM_SERVER_LOG_INFO << "process [" << req.process_name() << "] register preprocess func [" << req.extra_data() << "] for (" << req.old_mode() << "," << req.new_mode() << ") success";
            }
            SM_SERVER_LOG_INFO << "PreProcessFuncMap size is [" << PreProcessFuncMap.size() << "]";

            hozon::netaos::zmqipc::sm_reply resp{};
            resp.set_type(REPLY_CODE_REGISTER_PREPROCESS_FUNC);
            resp.set_process_name(GetProcessName());
            resp.set_result(0);
            SM_SERVER_LOG_INFO << "resp.set_type is " << FormatType(resp.type()) << ", resp.set_process_name() is [" << resp.process_name() << "], res is [" << resp.result() << "]";
            reply = resp.SerializeAsString();
            break;
        }
        case REQUEST_CODE_REGISTER_POSTPROCESS_FUNC:
        {
            SM_SERVER_LOG_INFO << "request process name is " << req.process_name() << "; old mode is " << req.old_mode() << "; new_mode is " << req.new_mode();
            {
                std::lock_guard<std::mutex> lock(postprocess_funcmap_mutex);
                auto &it = PostProcessFuncMap[make_pair(req.old_mode(), req.new_mode())];
                it.emplace(req.process_name(), req.extra_data());
                SM_SERVER_LOG_INFO << "process [" << req.process_name() << "] register postprocess func [" << req.extra_data() << "] for (" << req.old_mode() << "," << req.new_mode() << ") success";
            }
            SM_SERVER_LOG_INFO << "PostProcessFuncMap size is [" << PostProcessFuncMap.size() << "]";

            hozon::netaos::zmqipc::sm_reply resp{};
            resp.set_type(REPLY_CODE_REGISTER_POSTPROCESS_FUNC);
            resp.set_process_name(GetProcessName());
            resp.set_result(0);
            SM_SERVER_LOG_INFO << "resp.set_type is " << FormatType(resp.type()) << ", resp.set_process_name() is [" << resp.process_name() << "], res is [" << resp.result() << "]";
            reply = resp.SerializeAsString();
            break;
        }
        case REQUEST_CODE_SWITCH_MODE:
        {
            SM_SERVER_LOG_INFO << "SwitchMode Request";
            bool bResult = true;
            int32_t switchmode_result = -1;

            std::string process_name = req.process_name();
            std::string old_mode = ExecManagement::Instance()->GetSysMode();
            std::string new_mode = req.new_mode();
            SM_SERVER_LOG_INFO << "Request: process name is [" << process_name << "]; old_mode is [" << old_mode <<  "]; new_mode is [" << new_mode << "]";

            /* PreProcess */
            map<std::string, std::string> pre_trans_copy;
            {
                std::lock_guard<std::mutex> lock(preprocess_funcmap_mutex);
                SM_SERVER_LOG_INFO << "PreProcessFuncMap size is [" << PreProcessFuncMap.size() << "]";
                if (PreProcessFuncMap.count(make_pair(old_mode, new_mode)) > 0) {
                    auto pre_trans = PreProcessFuncMap[make_pair(old_mode, new_mode)];
                    copy(pre_trans.begin(), pre_trans.end(), inserter(pre_trans_copy, pre_trans_copy.begin()));
                    bResult = PreAndPostProcess(ProcessMode::PREPROCESS, old_mode, new_mode, pre_trans_copy);
                    if (!bResult) {
                        SM_SERVER_LOG_ERROR << "PreProcess failed";
                    }
                } else {
                    SM_SERVER_LOG_INFO << "PreProcessFuncMap has not element (" << old_mode << "," << new_mode << "), not need preprocess";
                }
            }

            /* SwitchMode */
            int32_t res = ExecManagement::Instance()->SwitchMode(new_mode);
            if (res == 0) {
                SM_SERVER_LOG_INFO << "ExecManagement::Instance()->SwitchMode to mode [" << new_mode << "] success";
                switchmode_result = static_cast<int32_t>(SmResultCode::kSuccess);
            } else {
                SM_SERVER_LOG_ERROR << "ExecManagement::Instance()->SwitchMode to mode [" << new_mode << "] failed, res is [" << res << "]";
                switchmode_result = res;
            }

            /* PostProcess */
            map<std::string, std::string> post_trans_copy;
            {
                std::lock_guard<std::mutex> lock(postprocess_funcmap_mutex);
                SM_SERVER_LOG_INFO << "PostProcessFuncMap size is [" << PostProcessFuncMap.size() << "]";
                if (PostProcessFuncMap.count(make_pair(old_mode, new_mode)) > 0) {
                    auto post_trans = PostProcessFuncMap[make_pair(old_mode, new_mode)];
                    copy(post_trans.begin(), post_trans.end(), inserter(post_trans_copy, post_trans_copy.begin()));
                    bResult = PreAndPostProcess(ProcessMode::POSTPROCESS, old_mode, new_mode, post_trans_copy);
                    if (!bResult) {
                        SM_SERVER_LOG_ERROR << "PostProcess failed";
                    }
                } else {
                    SM_SERVER_LOG_INFO << "PostProcessFuncMap has not element (" << old_mode << "," << new_mode << "), not need postprocess";
                }
            }

            hozon::netaos::zmqipc::sm_reply resp{};
            resp.set_type(REPLY_CODE_SWITCH_MODE);
            resp.set_process_name(GetProcessName());
            resp.set_result(switchmode_result);
            SM_SERVER_LOG_INFO << "resp.set_type is " << FormatType(resp.type()) << ", resp.set_process_name() is [" << resp.process_name() << "], res is [" << resp.result() << "]";
            reply = resp.SerializeAsString();
            break;
        }
        case REQUEST_CODE_STOP_MODE:
        {
            SM_SERVER_LOG_INFO << "StopMode Request, process name is " << req.process_name();

            int32_t res = ExecManagement::Instance()->StopMode();
            hozon::netaos::zmqipc::sm_reply resp{};
            resp.set_type(REPLY_CODE_STOP_MODE);
            resp.set_process_name(GetProcessName());
            resp.set_result(res);
            SM_SERVER_LOG_INFO << "resp.set_type is " << FormatType(resp.type()) << ", resp.set_process_name() is [" << resp.process_name() << "], res is [" << resp.result() << "]";
            reply = resp.SerializeAsString();
            break;
        }
        case REQUEST_CODE_GET_CURR_MODE:
        {
            SM_SERVER_LOG_DEBUG << "GetCurrMode Request, process name is " << req.process_name();

            hozon::netaos::zmqipc::sm_reply resp{};
            resp.set_type(REPLY_CODE_GET_CURR_MODE);
            resp.set_process_name(GetProcessName());

            std::string curr_mode = ExecManagement::Instance()->GetSysMode();
            SM_SERVER_LOG_DEBUG << "GetCurrMode Request, curr_mode is [" << curr_mode << "]";
            resp.set_curr_mode(curr_mode);
            resp.set_result(0);
            SM_SERVER_LOG_DEBUG << "resp.set_type is " << FormatType(resp.type()) << ", resp.set_process_name() is [" << resp.process_name() << "], res is [" << resp.result() << "]";
            reply = resp.SerializeAsString();
            break;
        }
        case REQUEST_CODE_SET_DEFAULT_MODE:
        {
            SM_SERVER_LOG_INFO << "SetDefaultMode Request, process name is " << req.process_name() << "; default mode is [" << req.new_mode() << "]";

            //调用ExecManagement的 SetDefaultMode
            int32_t res = ExecManagement::Instance()->SetDefaultMode(req.new_mode());
            SM_SERVER_LOG_INFO << "ExecManagement::Instance()->SetDefaultMode to mode [" << req.new_mode() << "], res is [" << res << "]";

            hozon::netaos::zmqipc::sm_reply resp{};
            resp.set_type(REPLY_CODE_SET_DEFAULT_MODE);
            resp.set_process_name(GetProcessName());
            resp.set_result(res);
            SM_SERVER_LOG_INFO << "resp.set_type is " << FormatType(resp.type()) << ", resp.set_process_name() is [" << resp.process_name() << "], res is [" << resp.result() << "]";
            reply = resp.SerializeAsString();
            break;
        }
        case REQUEST_CODE_GET_PROC_INFO:
        {
            SM_SERVER_LOG_DEBUG << "GetProcessInfo Request, process name is " << req.process_name();

            hozon::netaos::zmqipc::sm_reply resp{};
            resp.set_type(REPLY_CODE_GET_PROC_INFO);
            resp.set_process_name(GetProcessName());
            vector<ProcessInfo> proc_info_data;
            int32_t res = ExecManagement::Instance()->GetModeOfProcess(&proc_info_data);
            vector<hozon::netaos::zmqipc::process_info> reply_data;
            if (res == 0) {
                SM_SERVER_LOG_DEBUG << "GetProcessInfo Request, proc_info_data size is [" << proc_info_data.size() << "]";

                for (auto proc_info: proc_info_data) {
                    SM_SERVER_LOG_TRACE << "Copy ProcessInfo: " << proc_info.procname;
                    hozon::netaos::zmqipc::process_info proc_info_temp;
                    proc_info_temp.set_group(proc_info.group);
                    proc_info_temp.set_procname(proc_info.procname);
                    proc_info_temp.set_procstate(static_cast<uint32_t>(proc_info.procstate));
                    reply_data.emplace_back(proc_info_temp);
                }
            }
            for (auto it : reply_data) {
                resp.add_data()->CopyFrom(it);
            }
            resp.set_result(res);
            SM_SERVER_LOG_DEBUG << "resp.set_type is " << FormatType(resp.type()) << ", resp.set_process_name() is [" << resp.process_name() << "], res is [" << resp.result() << "]";
            reply = resp.SerializeAsString();
            break;
        }
        case REQUEST_CODE_PROC_RESTART:
        {
            SM_SERVER_LOG_INFO << "ProcRestart Request, process name is " << req.extra_data();

            hozon::netaos::zmqipc::sm_reply resp{};
            resp.set_type(REPLY_CODE_PROC_RESTART);
            resp.set_process_name(GetProcessName());

            int32_t res = ExecManagement::Instance()->ProcRestart(req.extra_data());
            resp.set_result(res);
            SM_SERVER_LOG_INFO << "resp.set_type is " << FormatType(resp.type()) << ", resp.set_process_name() is [" << resp.process_name() << "], res is [" << resp.result() << "]";
            reply = resp.SerializeAsString();
            break;
        }
        case REQUEST_CODE_GET_MODE_LIST:
        {
            SM_SERVER_LOG_INFO << "GetModeList Request";

            hozon::netaos::zmqipc::sm_reply resp{};
            resp.set_type(REPLY_CODE_GET_MODE_LIST);
            resp.set_process_name(GetProcessName());

            vector<std::string> mode_list;
            int32_t res = ExecManagement::Instance()->GetModeList(&mode_list);
            SM_SERVER_LOG_INFO << "GetModeList Request, mode_list size is [" << mode_list.size() << "]";

            for (auto it : mode_list) {
                resp.add_mode_list(it);
            }
            resp.set_result(res);
            SM_SERVER_LOG_INFO << "resp.set_type is " << FormatType(resp.type()) << ", resp.set_process_name() is [" << resp.process_name() << "], res is [" << resp.result() << "]";
            reply = resp.SerializeAsString();
            break;
        }
        case REQUEST_CODE_GET_MODE_LIST_DETAIL_INFO:
        {
            SM_SERVER_LOG_INFO << "GetModeListDetailInfo Request";
            
            std::unordered_map<std::string, std::vector<std::shared_ptr<Process>>> mode_name_process_list_map;
            int32_t res = ExecManagement::Instance()->GetModeListDetailInfo(mode_name_process_list_map);
            (void) res;
            SM_SERVER_LOG_INFO << "GetModeListDetailInfo Request, mode_name_process_list_map size is [" << mode_name_process_list_map.size() << "]";

            hozon::netaos::zmqipc::sm_reply resp;
            resp.set_type(REQUEST_CODE_GET_MODE_LIST_DETAIL_INFO);
            resp.set_process_name(GetProcessName());
            
            uint32_t order = 0;
            for (const auto &[mode, process_list] : mode_name_process_list_map) {
                for (const auto &process : process_list) {
                    auto process_info = resp.add_data();
                    process->GetOrderOfMode(mode, &order);
                    process_info->set_group(order);
                    process_info->set_procstate((uint32_t)process->m_proc_state);
                    process_info->set_procname(process->m_process_name);
                    process_info->set_mode_name(mode);
                }
            }
            
            resp.set_result(res);
            SM_SERVER_LOG_INFO  << "resp.set_type is " << FormatType(resp.type())
                                << ", resp.set_process_name() is ["
                                << resp.process_name() << "], res is ["
                                << resp.result() << "]";
            reply = resp.SerializeAsString();

            break;
        }
        default:
        {
            SM_SERVER_LOG_INFO << "type can't be recognized";
            break;
        }
    }
    return 0;
}

} // namespace sm
} // namespace netaos
} // namespace hozon
