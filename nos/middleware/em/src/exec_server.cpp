/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: em
 * Description: exec_server.cpp
 * Created on: Feb 7, 2023
 * Author: renhongyan
 * 
 */

#include "em/include/exec_server.h"
#include "em/include/exec_server_impl.h"
#include "em/include/exec_server_impl_zmq.h"

namespace hozon {
namespace netaos{
namespace em{

ExecServer::ExecServer() {
	_pimpl = std::make_shared<ExecServerImplZmq>();
}

ExecServer::~ExecServer() {

}

int32_t ExecServer::Start()
{
	return _pimpl->Start();
}

void ExecServer::Stop()
{
	_pimpl->Stop();
}

} // namespace em
} // namespace netaos
} // namespace hozon
