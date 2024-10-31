/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: em
 * Description: exec_client.h
 * Created on: Feb 7, 2023
 * Author: renhongyan
 * 
 */
#ifndef EXEC_CLIENT_H
#define EXEC_CLIENT_H

#include <functional>
#include <memory>
#include "em/include/proctypes.h"

namespace hozon {
namespace netaos{
namespace em{

class ExecClientImplZmq;
class ExecClient {
public:

    ExecClient();
    ~ExecClient();
	int32_t ReportState(ExecutionState State);

private:
	std::shared_ptr<ExecClientImplZmq> _pimpl;
};
} // namespace em
} // namespace netaos
} // namespace hozon
#endif
