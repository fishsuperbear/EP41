// HelloWorldClient.cpp
#include <unistd.h>

#include <iostream>
#include <string>

#include <CommonAPI/CommonAPI.hpp>
#include <v1/commonapi/DoSomeIPProxy.hpp>

using namespace v1::commonapi;

void print(DoSomeIP::DoSomeIPRespUdsMessage msg)
{
	std::cout << "UDSResp from server is : "<< "'\n";
    std::cout << "udsSa: " << msg.getUdsSa() << std::endl;
    std::cout << "udsTa: " << msg.getUdsTa() << std::endl;
    std::cout << "result: " << msg.getResult() << std::endl;
    std::cout << "taType: " << msg.getTaType().toString() << std::endl;
    std::vector<uint8_t> tmp = msg.getUdsData();
    std::cout << "udsData size is: " << tmp.size() << std::endl;
    std::cout << "udsData: ";
    for (const auto& value : tmp) {
        std::cout << std::hex << static_cast<int>(value) << " ";
    }
    std::cout << std::endl;
}

int main() {

    std::shared_ptr<CommonAPI::Runtime> runtime = CommonAPI::Runtime::get();

    std::string domain = "local";
    std::string instance = "commonapi.dosomeip";
    std::string connection = "client-sample";

    std::shared_ptr<DoSomeIPProxy<>> myProxy = runtime->buildProxy <DoSomeIPProxy> (domain, instance, connection);


    while (!myProxy->isAvailable()) {
        std::cout << "Not Available !!! " << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }

    std::cout << "Server Available !!! " << std::endl;

    while (true) {
	    DoSomeIP::DoSomeIPReqUdsMessage req{};
        uint16_t source_addr = 6666;
        uint16_t target_addr = 8888;
        DoSomeIP::TargetAddressType type = DoSomeIP::TargetAddressType::kFunctional;
        DoSomeIP::my_array array = {3,1,0,1,0x0f,0x0f,0,1};
        req.setUdsSa(source_addr);
        req.setUdsTa(target_addr);
        req.setTaType(type);
        req.setUdsData(array);

        CommonAPI::CallStatus callStatus;
        DoSomeIP::stdErrorTypeEnum methodError;
	    DoSomeIP::DoSomeIPRespUdsMessage resp{};

        // Synchronous call
        std::cout << "Call udsMessageRequest with synchronous semantics ..." << std::endl;
        myProxy->udsMessageRequest(req, callStatus, methodError, resp);

        std::cout << "Result of synchronous call of udsMessageRequest: " << std::endl;
        std::cout << "   callStatus: " << ((callStatus == CommonAPI::CallStatus::SUCCESS) ? "SUCCESS" : "NO_SUCCESS")
                  << std::endl;
        std::cout << "   error: "
                  << ((methodError == DoSomeIP::stdErrorTypeEnum::OK) ? "OK" : "NOT_OK")
                  << std::endl;
        // 打印
        print(resp);
        std::cout << "Call udsMessageRequest end !!!" << std::endl;

        std::this_thread::sleep_for(std::chrono::seconds(10));
    }

    return 0;
}
