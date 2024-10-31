/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: sm
 * Description: state_server_impl.h
 * Created on: Feb 7, 2023
 * Author: renhongyan
 * 
 */

#ifndef STATE_SERVER_IMPL_H
#define STATE_SERVER_IMPL_H
#include <mutex>
#include <map>
#include "cm/include/method.h"
#include "idl/generated/smPubSubTypes.h"
#include "sm/include/sm_types.h"
#include "sm/include/sm_logger.h"

using namespace std;
using namespace hozon::netaos::cm;
using namespace hozon::netaos::log;

namespace hozon {
namespace netaos{
namespace sm{

class StateServerImpl {
public:

    class StateCMServer:public Server<sm_request, sm_reply> {
        public:
        StateCMServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data, StateServerImpl*  server) : Server(req_data, resp_data),
        state_server(server){};
        ~StateCMServer(){};
        int32_t Process(const std::shared_ptr<sm_request> req, std::shared_ptr<sm_reply> resp) {
            SM_SERVER_LOG_TRACE << "StateCMServer callback: received, type is " << FormatType(req->type());
            return state_server->RequestProcess(req, resp);
        }
        private:
        StateServerImpl * state_server;
    };

    StateServerImpl(): state_cm_server(std::make_shared<sm_requestPubSubType>(), std::make_shared<sm_replyPubSubType>(), this),
        client(std::make_shared<sm_requestPubSubType>(), std::make_shared<sm_replyPubSubType>()){
    };
    ~StateServerImpl(){};
    int32_t Start();
    void Stop();
    int32_t RequestProcess(const std::shared_ptr<sm_request> req, std::shared_ptr<sm_reply> resp);

private:
    std::string GetProcessName();
    int32_t SendRequestToClient(const uint32_t& type, const uint32_t& reply_type, const std::string& old_mode, const std::string& new_mode, 
        const bool& succ=true, const string& client_process_name="",const string& extra_data="", const bool& forget=false);

    bool PreAndPostProcess(const ProcessMode mod, const std::string& src_mode, const std::string& tar_mode, std::map<string, string>& map);

    StateCMServer state_cm_server;
    map<pair<string,string>,map<string, string>> PreProcessFuncMap;
    map<pair<string,string>,map<string, string>> PostProcessFuncMap;
    std::mutex preprocess_funcmap_mutex;
    std::mutex postprocess_funcmap_mutex;
    std::shared_ptr<sm_requestPubSubType> req_data_type = std::make_shared<sm_requestPubSubType>();
    std::shared_ptr<sm_replyPubSubType> resp_data_type = std::make_shared<sm_replyPubSubType>();
    std::shared_ptr<sm_request> req_data = std::make_shared<sm_request>();
    std::shared_ptr<sm_reply> resp_data = std::make_shared<sm_reply>();

    Client<sm_request, sm_reply> client;

};
} // namespace sm
} // namespace netaos
} // namespace hozon
#endif