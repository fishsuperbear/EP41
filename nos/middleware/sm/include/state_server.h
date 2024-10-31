/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: sm
 * Description: state_server.h
 * Created on: Feb 7, 2023
 * Author: renhongyan
 * 
 */

#ifndef STATE_SERVER_H
#define STATE_SERVER_H
#include <memory>

namespace hozon {
namespace netaos{
namespace sm{

class StateServerImpl;
class StateServerImplZmq;
class StateServer {
public:

    StateServer();
    ~StateServer();
    int32_t Start();
    void Stop();

private:
    std::unique_ptr<StateServerImpl> _pimpl;
    std::unique_ptr<StateServerImplZmq> _pimplzmq;
};
} // namespace sm
} // namespace netaos
} // namespace hozon
#endif