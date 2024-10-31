// HelloWorldClient.cpp
#include <unistd.h>

#include <iostream>
#include <string>

#include <CommonAPI/CommonAPI.hpp>
#include <v1/commonapi/DoSomeIPProxy.hpp>

using namespace v1::commonapi;

std::shared_ptr<DoSomeIPProxy<>> myProxy = nullptr;

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

void getReqMessage(DoSomeIP::DoSomeIPReqUdsMessage& msg)
{
    DoSomeIP::DoSomeIPReqUdsMessage req{};
    uint16_t source_addr = 0x6666;
    uint16_t target_addr = 0x8888;
    DoSomeIP::TargetAddressType type = DoSomeIP::TargetAddressType::kFunctional;
    DoSomeIP::my_array array = {3,1,0,1,15,15,0,1};
    req.setUdsSa(source_addr);
    req.setUdsTa(target_addr);
    req.setTaType(type);
    req.setUdsData(array);
    msg = req;
}

void recv_callback(const CommonAPI::CallStatus& callStatus, const DoSomeIP::stdErrorTypeEnum& methodError, const DoSomeIP::DoSomeIPRespUdsMessage& resp)
{
    std::cout << "recv_callback!";
    std::cout << "Result of synchronous call of udsMessageRequest: " << std::endl;
    std::cout << "   callStatus: " << ((callStatus == CommonAPI::CallStatus::SUCCESS) ? "SUCCESS" : "NO_SUCCESS")
                << std::endl;
    std::cout << "   error: "
                << ((methodError == DoSomeIP::stdErrorTypeEnum::OK) ? "OK" : "NOT_OK")
                << std::endl;
    // 打印
    print(resp);
    std::cout << "Call udsMessageRequest end !!!" << std::endl;
}


// 同步单次调用
void syncCallOnce()
{
    DoSomeIP::DoSomeIPReqUdsMessage req{};
    getReqMessage(req);

    CommonAPI::CallStatus callStatus;
    DoSomeIP::stdErrorTypeEnum methodError;
    DoSomeIP::DoSomeIPRespUdsMessage resp{};
    CommonAPI::Timeout_t timeout = 30000;
    CommonAPI::CallInfo* info = new CommonAPI::CallInfo(timeout);

    // Synchronous call
    std::cout << "Call udsMessageRequest with synchronous semantics ..." << std::endl;
    myProxy->udsMessageRequest(req, callStatus, methodError, resp, info);

    std::cout << "Result of synchronous call of udsMessageRequest: " << std::endl;
    std::cout << "   callStatus: " << ((callStatus == CommonAPI::CallStatus::SUCCESS) ? "SUCCESS" : "NO_SUCCESS")
                << std::endl;
    std::cout << "   callStatus from dosomeip : " << static_cast<uint16_t>(callStatus) << std::endl;;
    
    std::cout << "   error: "
                << ((methodError == DoSomeIP::stdErrorTypeEnum::OK) ? "OK" : "NOT_OK")
                << std::endl;
    // 打印
    print(resp);
    std::cout << "Call udsMessageRequest end !!!" << std::endl;
}
// 同步多次调用
void syncCalls()
{
    while (true) {
        DoSomeIP::DoSomeIPReqUdsMessage req{};
        getReqMessage(req);

        CommonAPI::CallStatus callStatus;
        DoSomeIP::stdErrorTypeEnum methodError;
	    DoSomeIP::DoSomeIPRespUdsMessage resp{};
        CommonAPI::Timeout_t timeout = 30000;
        CommonAPI::CallInfo* info = new CommonAPI::CallInfo(timeout);

        // Synchronous call
        std::cout << "Call udsMessageRequest with synchronous semantics ..." << std::endl;
        myProxy->udsMessageRequest(req, callStatus, methodError, resp, info);

        std::cout << "Result of synchronous call of udsMessageRequest: " << std::endl;
        std::cout << "   callStatus: " << ((callStatus == CommonAPI::CallStatus::SUCCESS) ? "SUCCESS" : "NO_SUCCESS")
                  << std::endl;
        std::cout << "   callStatus from dosomeip : " << static_cast<uint16_t>(callStatus) << std::endl;;
        
        std::cout << "   error: "
                  << ((methodError == DoSomeIP::stdErrorTypeEnum::OK) ? "OK" : "NOT_OK")
                  << std::endl;
        // 打印
        print(resp);
        std::cout << "Call udsMessageRequest end !!!" << std::endl;

        std::this_thread::sleep_for(std::chrono::seconds(10));
    }
}
// 异步单次调用
void asyncCallOnce()
{
    std::function<
        void(const CommonAPI::CallStatus&, 
        const DoSomeIP::stdErrorTypeEnum&, 
        const DoSomeIP::DoSomeIPRespUdsMessage&)> 
    UdsMessageRequestAsyncCallback = recv_callback;
    DoSomeIP::DoSomeIPReqUdsMessage req{};
    CommonAPI::Timeout_t timeout = 30000;
    CommonAPI::CallInfo* info = new CommonAPI::CallInfo(timeout);
    getReqMessage(req);
    auto fut = myProxy->udsMessageRequestAsync(req, UdsMessageRequestAsyncCallback, info);

    std::cout << "waiting...\n";
    std::future_status status;
    do {
        status = fut.wait_for(std::chrono::seconds(10));
        if (status == std::future_status::deferred) {
            std::cout << "deferred\n";
        } else if (status == std::future_status::timeout) {
            std::cout << "timeout\n";
        } else if (status == std::future_status::ready) {
            std::cout << "ready!\n";
        }
    } while (status != std::future_status::ready); 
}
// 异步多次调用
void asyncCalls()
{
    while (true) {
        std::function<
            void(const CommonAPI::CallStatus&, 
            const DoSomeIP::stdErrorTypeEnum&, 
            const DoSomeIP::DoSomeIPRespUdsMessage&)> 
        UdsMessageRequestAsyncCallback = recv_callback;
        DoSomeIP::DoSomeIPReqUdsMessage req{};
        CommonAPI::Timeout_t timeout = 30000;
        CommonAPI::CallInfo* info = new CommonAPI::CallInfo(timeout);
        getReqMessage(req);
        auto fut = myProxy->udsMessageRequestAsync(req, UdsMessageRequestAsyncCallback, info);

        std::cout << "waiting...\n";
        std::future_status status;
        do {
            status = fut.wait_for(std::chrono::seconds(10));
            if (status == std::future_status::deferred) {
                std::cout << "deferred\n";
            } else if (status == std::future_status::timeout) {
                std::cout << "timeout\n";
            } else if (status == std::future_status::ready) {
                std::cout << "ready!\n";
            }
        } while (status != std::future_status::ready); 
        std::this_thread::sleep_for(std::chrono::seconds(10));
    }
}

void printHelp()
{
    std::cout << R"(
        用法: 
            ./someip_client [参数]      

        参数:
        sync			同步单次调用
        syncs			同步多次调用 
        async           异步单次调用
        asyncs          异步多次调用

        举例：
        ./someip_client sync
    )" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Please check argv. refer to the following help: " << std::endl;
        printHelp();
        return -1;
    }

    std::shared_ptr<CommonAPI::Runtime> runtime = CommonAPI::Runtime::get();

    std::string domain = "local";
    std::string instance = "commonapi.dosomeip";
    std::string connection = "client-sample";

    myProxy = runtime->buildProxy <DoSomeIPProxy> (domain, instance, connection);

    while (!myProxy->isAvailable()) {
        std::cout << "Not Available !!! " << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }

    std::cout << "Server Available !!! " << std::endl;

    std::string msg = argv[1];

    if (msg == "sync") {
        std::thread* thread01 = new std::thread(syncCallOnce);
        thread01->join();
    } else if (msg == "syncs") {
        std::thread* thread02 = new std::thread(syncCalls);
        thread02->join();
    } else if (msg == "async") {
        std::thread* thread03 = new std::thread(asyncCallOnce);
        thread03->join();
    } else if (msg == "asyncs") {
        std::thread* thread04 = new std::thread(syncCalls);
        thread04->join();
    } else {
        printHelp();
    }

    return 0;
}
