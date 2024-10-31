/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: em
 * Description: exec_client_impl_zmq.h
 * Created on: Nov 22, 2023
 * Author: xumengjun
 * 
 */
#ifndef STATE_CLIENT_IMPL_ZMQ_H
#define STATE_CLIENT_IMPL_ZMQ_H
#include <map>

#include "idl/generated/emPubSubTypes.h"
#include "idl/generated/em.h"
#include "em/include/lblogger.h"
#include "em/include/proctypes.h"
#include "zmq_ipc/manager/zmq_ipc_client.h"

namespace hozon {
namespace netaos{
namespace em{

class ExecClientImplZmq {
public:
    ExecClientImplZmq();
    ~ExecClientImplZmq();
    int32_t ReportState(ExecutionState state);

private:
	bool GetProcessName(std::string &process_name);
	int32_t SendRequestToServer(const uint32_t& type, const int32_t state, const std::string& target_process_name="");

private:
    std::shared_ptr<hozon::netaos::zmqipc::ZmqIpcClient> client_zmq_;

    const std::string em_client_name_ = "tcp://localhost:11151";
};
} // namespace em
} // namespace netaos
} // namespace hozon
#endif
