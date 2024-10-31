/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: em
 * Description: exec_client.cpp
 * Created on: Feb 7, 2023
 * Author: renhongyan
 *
 */
#include "em/include/proctypes.h"
#include "em/include/exec_client.h"
#include "em/include/exec_client_impl.h"
#include "em/include/exec_client_impl_zmq.h"

namespace hozon {
namespace netaos{
namespace em{

ExecClient::ExecClient()
{
	_pimpl = std::make_shared<ExecClientImplZmq>();
}

ExecClient::~ExecClient() {
}

int32_t ExecClient::ReportState(ExecutionState State)
{
	return _pimpl->ReportState(State);
}


} // namespace em
} // namespace netaos
} // namespace hozon
