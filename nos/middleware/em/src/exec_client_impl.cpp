/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: em
 * Description: exec_client_impl.cpp
 * Created on: Feb 7, 2023
 * Author: renhongyan
 *
 */
#include "em/include/proctypes.h"
#include "em/include/exec_client_impl.h"
#include <iostream>

using namespace std;
using namespace hozon::netaos::log;
extern char **environ;

namespace hozon {
namespace netaos{
namespace em{

ExecClientImpl::ExecClientImpl():
	req_data_type(std::make_shared<em_requestPubSubType>()),
    resp_data_type(std::make_shared<em_replyPubSubType>()),
    req_data(std::make_shared<em_request>()),
    resp_data(std::make_shared<em_reply>()),
    client(std::make_shared<em_requestPubSubType>(), std::make_shared<em_replyPubSubType>())
{
	client.Init(0, "exec_client_request");
	client.WaitServiceOnline(1000);  //用户需要去调等待服务
}

ExecClientImpl::~ExecClientImpl() {
	LB_LOG_INFO <<  "ExecClientImpl.Deinit()";
	client.Deinit();
}

int32_t ExecClientImpl::SendRequestToServer(const uint32_t& type, const int32_t state, const std::string& target_process_name)
{
    std::string process_name;
    if (!GetProcessName(process_name)) {
        LB_LOG_INFO << "SendRequestToServer: get process name failed, don't send request to server, environ[0]:" << process_name;
        return 0;
    }

	LB_LOG_INFO << "SendRequestToServer: type = [" << LogHex32{type} << "] ExecutionState is [" << (int)state << "]";
	req_data->type(type);
	req_data->process_name(process_name);
	req_data->target_process_name(target_process_name);
	req_data->state(state);
    int res = client.Request(req_data, resp_data, 5000);
	if (res != 0) {
		LB_LOG_INFO << "SendRequestToServer: client.Request result is NG, res is " << res<< ";process name is ["<< process_name <<"]";
		res = static_cast<int32_t>(ResultCode::kTimeout);
	}
	else {
		if (resp_data->result() == 0) {
			LB_LOG_INFO << "Request " << LogHex32{type} << ", reply type " << LogHex32{resp_data->type()} << ", is OK, and result[" << resp_data->result() << "] is success";
		} else {
			LB_LOG_INFO << "Request " << LogHex32{type} << ", reply type " << LogHex32{resp_data->type()} << ", is OK, but result[" << resp_data->result() << "] is fail";
		}
		res = resp_data->result();
	}
	return res;
}

int32_t ExecClientImpl::ReportState(ExecutionState state)
{
    LB_LOG_INFO << "ExecClientImpl::ReportState is called";
    client.WaitServiceOnline(3000);
    if (state == ExecutionState::kTerminating) {
        LB_LOG_INFO << "client deinit";
        client.Deinit();
    }
    return SendRequestToServer(REQUEST_CODE_REPORT_STATE, (int32_t)state);
}

bool ExecClientImpl::GetProcessName(std::string &process_name)
{
    bool res = true;
    process_name.assign(environ[0], strlen(environ[0]));
    size_t pos = process_name.find(ENVRION_NAME);
    if (pos != process_name.npos) {
        process_name.replace(pos, strlen(ENVRION_NAME), "");
    }else{
        LB_LOG_DEBUG<<"< get environ proc name fail! >";
        res = false;
    }

    return res;
}


} // namespace em
} // namespace netaos
} // namespace hozon
