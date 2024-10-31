// HelloWorldClient.cpp
#include <unistd.h>

#include <iostream>
#include <string>

#include <CommonAPI/CommonAPI.hpp>
#include <v1/commonapi/HelloWorldProxy.hpp>

using namespace v1_0::commonapi;

int main() {
    std::shared_ptr<CommonAPI::Runtime> runtime = CommonAPI::Runtime::get();
    std::shared_ptr<HelloWorldProxy<>> myProxy = runtime->buildProxy<HelloWorldProxy>("local", "test", "client-sample");

    std::cout << "Checking availability!" << std::endl;
    while (!myProxy->isAvailable()) usleep(10);
    std::cout << "Available..." << std::endl;

    // Subscribe to broadcast
    myProxy->getMcuCANMsgServiceEvent().subscribe([&](const v1_0::commonapi::HelloWorld::McuCANMsgAlgo& val) { std::cout << "Received status event: " << val.getFD6_ID4_axle_X() << std::endl; });

    while (true) {
        // Asynchronous call
        std::cout << "Call foo with asynchronous semantics ..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }

    return 0;
}
