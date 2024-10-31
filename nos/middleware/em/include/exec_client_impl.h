/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: em
 * Description: exec_client_impl.h
 * Created on: Feb 7, 2023
 * Author: renhongyan
 * 
 */
#ifndef STATE_CLIENT_IMPL_H
#define STATE_CLIENT_IMPL_H
#include <map>
#include "cm/include/method.h"
#include "idl/generated/emPubSubTypes.h"
#include "idl/generated/em.h"
#include "em/include/lblogger.h"

using namespace std;
using namespace hozon::netaos::cm;

namespace hozon {
namespace netaos{
namespace em{

class ExecClientImpl {
public:
    ExecClientImpl();
    ~ExecClientImpl();
    int32_t ReportState(ExecutionState state);

private:
	bool GetProcessName(std::string &process_name);
	int32_t SendRequestToServer(const uint32_t& type, const int32_t state, const std::string& target_process_name="");

private:
	std::shared_ptr<em_requestPubSubType> req_data_type;
    std::shared_ptr<em_replyPubSubType> resp_data_type;
    std::shared_ptr<em_request> req_data;
    std::shared_ptr<em_reply> resp_data;
    Client<em_request, em_reply> client;
};
} // namespace em
} // namespace netaos
} // namespace hozon
#endif
