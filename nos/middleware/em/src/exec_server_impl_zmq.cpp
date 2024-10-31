/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: em
 * Description: exec_server_impl_zmq.cpp
 * Created on: Nov 22, 2023
 * Author: xumengjun
 *
 */

#include "em/include/exec_server_impl_zmq.h"

#include "zmq_ipc/proto/em.pb.h"

#include "em/include/proctypes.h"
#include "em/include/execmanagement.h"
#include "em/include/process.h"

using namespace std;
using namespace hozon::netaos::em;
namespace hozon {
namespace netaos{
namespace em{

std::string ExecServerImplZmq::GetProcessName()
{
	return "ExecServerImplZmq";
}

int32_t ExecServerImplZmq::Start()
{
	LOG_INFO << "exec_zmq_server Start with " << em_service_name_;
    exec_zmq_server.Start(em_service_name_);
	return 0;
}

void ExecServerImplZmq::Stop()
{
    exec_zmq_server.Stop();
}

int32_t ExecServerImplZmq::RequestProcess(const std::string& request, std::string& reply) {
	LOG_INFO << "ExecServerImplZmq::RequestProcess received";

    hozon::netaos::zmqipc::em_request req;
    req.ParseFromString(request);
	switch(req.type()) {
		case REQUEST_CODE_REPORT_STATE:
		{
			LOG_INFO << "recv process " << req.process_name()<< " report state is " << exec_state[(int32_t)req.state()];

			std::shared_ptr<Process> ptr = ExecManagement::Instance()->GetProcess(req.process_name());
			if (ptr != nullptr) {
				ptr->SetExecState(static_cast<ExecutionState>(req.state()));
			}else{
				LOG_INFO <<"failed to get "<<req.process_name()<<" handle";
			}

            hozon::netaos::zmqipc::em_reply resp;
            resp.set_type(REPLY_CODE_REPORT_STATE);
            resp.set_process_name(GetProcessName());
            resp.set_result(0);
            reply = resp.SerializeAsString();

			break;
		}

		default:
		{
			LOG_INFO << "type can't be recognized";
			return -1;
		}

	}
	return 0;
}

} // namespace em
} // namespace netaos
} // namespace hozon
