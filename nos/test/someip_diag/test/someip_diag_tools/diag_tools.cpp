// HelloWorldClient.cpp
#include <unistd.h>

#include <iostream>
#include <string>

#include <CommonAPI/CommonAPI.hpp>
#include <v1/commonapi/DoSomeIPProxy.hpp>

#include "someip_config.h"

using namespace v1::commonapi;

std::shared_ptr<DoSomeIPProxy<>> myProxy = nullptr;

size_t req_count = 1;

std::uint16_t getID()
{
    return req_count;
}

void print(DoSomeIP::DoSomeIPRespUdsMessage msg)
{
	std::cout << "UDSResp from server is : "<< std::endl;
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

bool getReqMessage(uint16_t id, DoSomeIP::DoSomeIPReqUdsMessage& msg)
{

    DoSomeIP::DoSomeIPReqUdsMessage req{};
    uint16_t source_addr = SomeIPConfig::Instance()->getSourceAdr();
    std::cout << "s_address :  "<< source_addr << std::endl;

    uint16_t target_addr = SomeIPConfig::Instance()->getTargetAdr();
    std::cout << "t_address :  "<< target_addr << std::endl;

    DoSomeIP::TargetAddressType type = DoSomeIP::TargetAddressType::kPhysical;
    // 获取UDS命令，如果为空，说明已经执行完成。
    auto data = SomeIPConfig::Instance()->getUdsReqById(id); 

    if (data.size() == 0)
    {
        return false;
    }
    DoSomeIP::my_array array =  data;
    req.setUdsSa(source_addr);
    req.setUdsTa(target_addr);
    req.setTaType(type);
    req.setUdsData(array);
    msg = req;
    return true;
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
    if (resp.getResult() == 0)
    {
        SomeIPConfig::Instance()->setRespResultById(getID(), true);
    } else {
        SomeIPConfig::Instance()->setRespResultById(getID(), false);
    }
    std::cout << "Call udsMessageRequest end !!!" << std::endl;
}

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
    
    while (true)
    {
        auto res = getReqMessage(req_count, req);
        if (!res)
        {
            std::cout << "udsMessageRequest end, size is: " << req_count << std::endl;
            break;
        } else {
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
        req_count++;
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    std::cout << "end!!!\n";
    SomeIPConfig::Instance()->printUdsReqs();
}

int main() {
    std::string domain = "local";
    std::string instance = "commonapi.dosomeip";
    std::string connection = "client-sample";

    SomeIPConfig::Instance()->LoadConfig();
    SomeIPConfig::Instance()->printUdsReqs();
    std::shared_ptr<CommonAPI::Runtime> runtime = CommonAPI::Runtime::get();

    myProxy = runtime->buildProxy <DoSomeIPProxy> (domain, instance, connection);

    while (!myProxy->isAvailable()) {
        std::cout << "Not Available !!! " << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
    std::cout << "Server Available !!! " << std::endl;

    std::thread* thread01 = new std::thread(asyncCallOnce);
    thread01->join();

    return 0;
}
