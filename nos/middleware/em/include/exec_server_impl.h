/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: em
 * Description: exec_server_impl.h
 * Created on: Feb 7, 2023
 * Author: renhongyan
 * 
 */

#ifndef EXEC_SERVER_IMPL_H
#define EXEC_SERVER_IMPL_H
#include <map>
#include "cm/include/method.h"
#include "idl/generated/emPubSubTypes.h"
#include "em/include/logger.h"

using namespace std;	
using namespace hozon::netaos::cm;

namespace hozon {
namespace netaos{
namespace em{

class ExecServerImpl {
public:

	class ExecCMServer:public Server<em_request, em_reply> {
		public:
		ExecCMServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data, ExecServerImpl*  server) : Server(req_data, resp_data),
		exec_server(server){};
		~ExecCMServer(){};
		int32_t Process(const std::shared_ptr<em_request> req, std::shared_ptr<em_reply> resp) {
			LOG_INFO << "ExecCMServer callback: received, type is " << req->type();
			return exec_server->RequestProcess(req, resp);
		}
		private:
		ExecServerImpl * exec_server;
	};

    ExecServerImpl(): exec_cm_server(std::make_shared<em_requestPubSubType>(), std::make_shared<em_replyPubSubType>(), this),
		client(std::make_shared<em_requestPubSubType>(), std::make_shared<em_replyPubSubType>()){
	};
    ~ExecServerImpl(){};
	int32_t Start();
	void Stop();
	int32_t RequestProcess(const std::shared_ptr<em_request> req, std::shared_ptr<em_reply> resp);
	
private:
	ExecCMServer exec_cm_server;

	std::string GetProcessName();
	std::shared_ptr<em_requestPubSubType> req_data_type = std::make_shared<em_requestPubSubType>();
    std::shared_ptr<em_replyPubSubType> resp_data_type = std::make_shared<em_replyPubSubType>();
    std::shared_ptr<em_request> req_data = std::make_shared<em_request>();
    std::shared_ptr<em_reply> resp_data = std::make_shared<em_reply>();

    Client<em_request, em_reply> client;

};
} // namespace em
} // namespace netaos
} // namespace hozon
#endif