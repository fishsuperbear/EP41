#include <unistd.h>

#include <cstddef>
#include <iostream>
#include <memory>
#include <thread>

#include "idl/generated/avm_methodPubSubTypes.h"
#include "idl/generated/chassis_methodPubSubTypes.h"
#include "log/include/default_logger.h"
#include "cm/include/method.h"

using namespace hozon::netaos::cm;

int main(int argc, char** argv) {
    DefaultLogger::GetInstance().InitLogger();
    std::shared_ptr<avm_methodPubSubType> req_data_type = std::make_shared<avm_methodPubSubType>();
    std::shared_ptr<chassis_methodPubSubType> resp_data_type = std::make_shared<chassis_methodPubSubType>();
    std::shared_ptr<avm_method> req_data = std::make_shared<avm_method>();
    std::shared_ptr<chassis_method> resq_data = std::make_shared<chassis_method>();

    Client<avm_method, chassis_method> client(req_data_type, resp_data_type);
    client.Init(2, "test");
    int online = client.WaitServiceOnline(2000);  //用户需要去调等待服务
    std::cout << "online: " << online << std::endl;

    // 0.Request
    req_data->fire_forget(false);
    client.Request(req_data, resq_data, 501);

    // // 1.Request
    std::thread task1([&req_data, &resq_data, &client] { client.Request(req_data, resq_data, 502); });
    std::thread task2([&req_data, &resq_data, &client] { client.Request(req_data, resq_data, 503); });


    // 2.AsyncRequest
    auto future = client.AsyncRequest(req_data, 504);
    future.wait();
    auto future_get = future.get();

    if (future_get.second != nullptr) {
        std::cout << "first: " << future_get.first << std::endl;             // first 不为0是客户端发生了错误
        std::cout << "second: " << future_get.second->reply() << std::endl;  // second不为0是服务端发生了错误
    } else {
        std::cout << "future.get().second == nullptr" << std::endl;
    }

    //for cm method testcase
    if((future_get.first == 0) && (future_get.second->reply() == 0) && (online == 0)){
        std::cout<<"!!!!!hz_test_success!!!!!"<<std::endl;
    }
    else{
        std::cout<<"!!!!!hz_test_failed!!!!!"<<std::endl;
    }

    // 3.Fire Forget
    req_data->fire_forget(false);
    client.Request(req_data, resq_data, 505);
    // client.RequestAndForget(req_data);

    if (task1.joinable()) {
        task1.join();
    }
    if (task2.joinable()) {
        task2.join();
    }

    for (std::size_t i = 0; i < 10; i++) {
        std::shared_ptr<avm_methodPubSubType> req_data_type = std::make_shared<avm_methodPubSubType>();
        std::shared_ptr<chassis_methodPubSubType> resp_data_type = std::make_shared<chassis_methodPubSubType>();
        std::shared_ptr<avm_method> req_data = std::make_shared<avm_method>();
        std::shared_ptr<chassis_method> resq_data = std::make_shared<chassis_method>();
        Client<avm_method, chassis_method> client_tmp(req_data_type, resp_data_type);
        client_tmp.Init(2, "test");
        int online = client_tmp.WaitServiceOnline(2000);  //用户需要去调等待服务
        client_tmp.Request(req_data, resq_data, 501);
        client_tmp.Deinit();
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    client.Deinit();

    return 0;
}
