/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: main function definition
 */

#ifndef DEVM_CLIENT_CM_METHOD_H_
#define DEVM_CLIENT_CM_METHOD_H_

#include "cm/include/method.h"
#include "idl/generated/devmPubSubTypes.h"


namespace hozon {
namespace netaos {
namespace devm_client {

using namespace hozon::netaos::cm;


class DevmServerMethod  : public Server<DevmReadDid, DevmReadDid> {
public:
    DevmServerMethod(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data) : Server(req_data, resp_data) {}
    virtual int32_t Process(const std::shared_ptr<DevmReadDid> req, std::shared_ptr<DevmReadDid> resp) {
        std::cout << "=========== process did " << req->did() << std::endl;
        return 0;
    }

    virtual ~DevmServerMethod() {}
private:
    // std::shared_ptr<DevmDataGathering> devm_data_gather_;
};




class DevmServerTransportMethod {
public:
    DevmServerTransportMethod() {
        method_server_ = nullptr;
    }
    ~DevmServerTransportMethod() {

    }

    void Init() {

        std::cout << "DevmServerTransportMethod::Init" << std::endl;
        std::shared_ptr<DevmReadDidPubSubType> req_data_type = std::make_shared<DevmReadDidPubSubType>();
        std::shared_ptr<DevmReadDidPubSubType> resp_data_type = std::make_shared<DevmReadDidPubSubType>();
        method_server_ = std::make_shared<DevmServerMethod>(req_data_type, resp_data_type);
        // method_server_->RegisterProcess(std::bind(&DevmServerMethod::Process, method_server_, std::placeholders::_1, std::placeholders::_2));
        method_server_->Start(0, "devm_read_did_tpl");
    }
    void DeInit() {

        std::cout << "DevmServerTransportMethod::DeInit" << std::endl;
        if (nullptr != method_server_) {
            method_server_->Stop();
            method_server_ = nullptr;
        }
    }

private:
    std::shared_ptr<DevmServerMethod> method_server_;

};


}  // namespace devm_client
}  // namespace netaos
}  // namespace hozon
#endif  // DEVM_CLIENT_CM_METHOD_H_


