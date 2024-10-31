// // HelloWorldStubImpl.cpp
#include "HelloWorldStubImpl.hpp"

HelloWorldStubImpl::HelloWorldStubImpl() {}
HelloWorldStubImpl::~HelloWorldStubImpl() {}

void HelloWorldStubImpl::incCounter(const v1::commonapi::HelloWorld::McuCANMsgAlgo& _test) {
    fireMcuCANMsgServiceEvent(_test);
    std::cout << "Send 123123123312 !!!!!!!!!" << std::endl;
}