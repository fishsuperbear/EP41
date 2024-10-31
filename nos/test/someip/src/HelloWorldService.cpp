// // HelloWorldService.cpp
#include <iostream>
#include <thread>

#include <CommonAPI/CommonAPI.hpp>

#include "HelloWorldStubImpl.hpp"

using namespace std;

int main() {
    std::shared_ptr<CommonAPI::Runtime> runtime = CommonAPI::Runtime::get();
    std::shared_ptr<HelloWorldStubImpl> myService = std::make_shared<HelloWorldStubImpl>();
    runtime->registerService("local", "test", myService, "service-sample");
    std::cout << "Successfully Registered Service!" << std::endl;
    v1::commonapi::HelloWorld::McuCANMsgAlgo _test;
    _test.setFD6_ID4_axle_X(12);
    while (true) {
        myService->incCounter(_test);
        std::cout << "Waiting for calls... (Abort with CTRL+C)" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(10));
    }
    return 0;
}