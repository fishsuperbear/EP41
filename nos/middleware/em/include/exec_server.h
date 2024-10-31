/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: em
 * Description: exec_server.h
 * Created on: Feb 7, 2023
 * Author: renhongyan
 * 
 */

#ifndef EXEC_SERVER_H
#define EXEC_SERVER_H
#include <memory>

namespace hozon {
namespace netaos{
namespace em{

class ExecServerImpl;
class ExecServerImplZmq;
class ExecServer {
public:

    ExecServer();
    ~ExecServer();
	int32_t Start();
	void Stop();
	
private:
	std::shared_ptr<ExecServerImplZmq> _pimpl;
};
} // namespace em
} // namespace netaos
} // namespace hozon
#endif
