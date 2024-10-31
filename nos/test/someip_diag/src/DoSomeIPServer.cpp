// Copyright (C) 2014-2019 Bayerische Motoren Werke Aktiengesellschaft (BMW AG)
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <thread>

#include <CommonAPI/CommonAPI.hpp>
#include "DoSomeIPStubImpl.hpp"

using namespace std;

int main() {

    std::shared_ptr<CommonAPI::Runtime> runtime = CommonAPI::Runtime::get();

    std::string domain = "local";
    std::string instance = "commonapi.dosomeip";
    std::string connection = "service-sample";

    std::shared_ptr<DoSomeIPStubImpl> myService = std::make_shared<DoSomeIPStubImpl>();
    bool successfullyRegistered = runtime->registerService(domain, instance, myService, connection);

    while (!successfullyRegistered) {
        std::cout << "Register Service failed, trying again in 5 seconds..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(5));
        successfullyRegistered = runtime->registerService(domain, instance, myService, connection);
    }

    std::cout << "Successfully Registered Service!" << std::endl;

    while (true) {
        std::cout << "Waiting for calls... (Abort with CTRL+C)" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(20));
    }

    return 0;
}
