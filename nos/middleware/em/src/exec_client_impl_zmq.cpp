/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: em
 * Description: exec_client_impl_zmq.cpp
 * Created on: Nov 22, 2023
 * Author: xumengjun
 *
 */
#include "em/include/exec_client_impl_zmq.h"

#include <iostream>

#include "zmq_ipc/proto/em.pb.h"

#include "em/include/proctypes.h"

using namespace std;
using namespace hozon::netaos::log;
extern char **environ;

namespace hozon {
namespace netaos{
namespace em{

ExecClientImplZmq::ExecClientImplZmq() {
    client_zmq_ = std::make_shared<hozon::netaos::zmqipc::ZmqIpcClient>();
    client_zmq_->Init(em_client_name_);
}

ExecClientImplZmq::~ExecClientImplZmq() {
	LB_LOG_INFO <<  "ExecClientImplZmq.Deinit()";
    client_zmq_->Deinit();
}

int32_t ExecClientImplZmq::SendRequestToServer(const uint32_t& type, const int32_t state, const std::string& target_process_name)
{
    std::string process_name;
    if (!GetProcessName(process_name)) {
        LB_LOG_INFO << "SendRequestToServer: get process name failed, don't send request to server, environ[0]:" << process_name;
        return 0;
    }

	LB_LOG_INFO << "SendRequestToServer: type = [" << LogHex32{type} << "] ExecutionState is [" << (int)state << "]";

    hozon::netaos::zmqipc::em_request req_data;
    req_data.set_type(type);
    req_data.set_process_name(process_name);
    req_data.set_target_process_name(target_process_name);
    req_data.set_state(state);

    std::string resp_data;
    int32_t res = client_zmq_->Request(req_data.SerializeAsString(), resp_data, 10000);
	if (res != 0) {
		LB_LOG_INFO << "SendRequestToServer: client_zmq.Request result is NG, res is " << res<< ";process name is ["<< process_name <<"]";
		res = static_cast<int32_t>(ResultCode::kTimeout);
    }
    else {
        hozon::netaos::zmqipc::em_reply reply;
        reply.ParseFromString(resp_data);

		if (reply.result() == 0) {
			LB_LOG_INFO << "Request " << LogHex32{type} << ", reply type "
                        << LogHex32{reply.type()} << ", is OK, and result[" << reply.result() << "] is success";
		} else {
			LB_LOG_INFO << "Request " << LogHex32{type} << ", reply type "
                        << LogHex32{reply.type()} << ", is OK, but result[" << reply.result() << "] is fail";
		}
		res = reply.result();
	}
	return res;
}

int32_t ExecClientImplZmq::ReportState(ExecutionState state)
{
	LB_LOG_INFO << "ExecClientImplZmq::ReportState is called";
	return SendRequestToServer(REQUEST_CODE_REPORT_STATE, (int32_t)state);
}

bool ExecClientImplZmq::GetProcessName(std::string &process_name)
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
