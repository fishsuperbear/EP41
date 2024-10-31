/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: sm
 * Description: state_client_impl.h
 * Created on: Feb 7, 2023
 * Author: renhongyan
 * 
 */
#ifndef STATE_CLIENT_IMPL_H
#define STATE_CLIENT_IMPL_H
#ifdef UT
#define private public
#define protected public
#endif //UT
#include <map>
#include "cm/include/method.h"
#include "idl/generated/smPubSubTypes.h"
#include "idl/generated/sm.h"
#include "sm/include/sm_types.h"
#include "sm/include/sm_logger.h"
#include "em/include/proctypes.h"

using namespace std;
using namespace hozon::netaos::cm;
using namespace hozon::netaos::em;
using namespace hozon::netaos::log;

namespace hozon {
namespace netaos{
namespace sm{

using PreProcessFunc = std::function<int32_t(const std::string& old_mode, const std::string& new_mode)>;
using PostProcessFunc = std::function<void(const std::string& old_mode, const std::string& new_mode, const bool succ)>;

class StateClientImpl {
public:

    class ProcessFuncServer:public Server<sm_request, sm_reply> {
        public:
        ProcessFuncServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data, StateClientImpl &client) : Server(req_data, resp_data),
        state_client(client){};
        ~ProcessFuncServer(){};
        int32_t Process(const std::shared_ptr<sm_request> req, std::shared_ptr<sm_reply> resp) {
            SM_CLIENT_LOG_INFO << "ProcessFuncServer callback: received, type is " << FormatType(req->type());
            return state_client.RequestProcess(req, resp);
        }
        private:
            StateClientImpl &state_client;
    };

    StateClientImpl();
    ~StateClientImpl();
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
    int32_t RequestProcess(const std::shared_ptr<sm_request> req, std::shared_ptr<sm_reply> resp);
    void SetProcessName(const std::string &proc_name) {processname_fordebug = proc_name;}
    std::string GetProcessName();

private:
    std::string processname_fordebug;
    int32_t SendRequestToServer(const uint32_t& type, const uint32_t& reply_type, const std::string& old_mode="", const std::string& new_mode="", const bool& succ=true, const string& extra_data="");
private:
    map<string,PreProcessFunc> PreProcessFuncMap;
    map<string,PostProcessFunc> PostProcessFuncMap;
    std::shared_ptr<sm_requestPubSubType> req_data_type;
    std::shared_ptr<sm_replyPubSubType> resp_data_type;
    std::shared_ptr<sm_request> req_data;
    std::shared_ptr<sm_reply> resp_data;
    Client<sm_request, sm_reply> client;

    ProcessFuncServer process_func_server;

};
} // namespace sm
} // namespace netaos
} // namespace hozon
#endif