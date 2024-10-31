// // HelloWorldStubImpl.hpp
#ifndef HELLOWORLDSTUBIMPL_H_
#define HELLOWORLDSTUBIMPL_H_
#include <CommonAPI/CommonAPI.hpp>
#include <v1/commonapi/HelloWorldStubDefault.hpp>

class HelloWorldStubImpl : public v1_0::commonapi::HelloWorldStubDefault {
   public:
    HelloWorldStubImpl();
    virtual ~HelloWorldStubImpl();
    virtual void incCounter(const v1::commonapi::HelloWorld::McuCANMsgAlgo& _test);
};
#endif
/* HELLOWORLDSTUBIMPL_H_ */
