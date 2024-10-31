/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: em
 * Description: exec_server_impl.cpp
 * Created on: Feb 7, 2023
 * Author: renhongyan
 * 
 */

#include "em/include/proctypes.h"
#include "em/include/exec_server_impl.h"
#include "em/include/execmanagement.h"
#include "em/include/process.h"

using namespace std;
using namespace hozon::netaos::em;
namespace hozon {
namespace netaos{
namespace em{

std::string ExecServerImpl::GetProcessName()
{
	return "ExecServerImpl";
}

int32_t ExecServerImpl::Start()
{
	LOG_INFO << "exec_cm_server Start with topic exec_client_request";
    exec_cm_server.Start(0, "exec_client_request");
	return 0;
}

void ExecServerImpl::Stop()	
{
    exec_cm_server.Stop();
}

int32_t ExecServerImpl::RequestProcess(const std::shared_ptr<em_request> req, std::shared_ptr<em_reply> resp)
{
	LOG_INFO << "ExecServerImpl::RequestProcess received";

	switch(req->type()) {
		case REQUEST_CODE_REPORT_STATE:
		{
			LOG_INFO << "request process name is " << req->process_name()<< "; state is " << (int32_t)req->state();
			
			std::shared_ptr<Process> ptr = ExecManagement::Instance()->GetProcess(req->process_name());
			if (ptr != nullptr) {
				ptr->SetExecState(static_cast<ExecutionState>(req->state()));
			}else{
				LOG_INFO <<"failed to get "<<req->process_name()<<" handle";
			}
			
			resp->type(REPLY_CODE_REPORT_STATE);
			resp->process_name(GetProcessName());
			resp->result(0);
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
