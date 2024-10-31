/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: sm
 * Description: state_server.cpp
 * Created on: Feb 7, 2023
 * Author: renhongyan
 * 
 */

#include "sm/include/state_server.h"
#include "sm/include/state_server_impl.h"
#include "sm/include/state_server_impl_zmq.h"

namespace hozon {
namespace netaos{
namespace sm{

StateServer::StateServer() {
    _pimpl = std::make_unique<StateServerImpl>();
    _pimplzmq = std::make_unique<StateServerImplZmq>();
}

StateServer::~StateServer() {

}

int32_t StateServer::Start()
{
    _pimplzmq->Start();
    return _pimpl->Start();
}

void StateServer::Stop()
{
    _pimplzmq->Stop();
    _pimpl->Stop();
}

} // namespace sm
} // namespace netaos
} // namespace hozon
