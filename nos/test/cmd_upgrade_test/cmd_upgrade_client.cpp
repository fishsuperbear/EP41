#include <iostream>
#include "cm/include/method.h"
#include "idl/generated/devmPubSubTypes.h"

using namespace hozon::netaos::cm;

void testCaseUpdateStatus()
{
    std::shared_ptr<Client<common_req, update_status_resp>> client_;
    std::shared_ptr<common_reqPubSubType> req_type = std::make_shared<common_reqPubSubType>();
    std::shared_ptr<update_status_respPubSubType> resp_type = std::make_shared<update_status_respPubSubType>();
    client_ = std::make_shared<Client<common_req, update_status_resp>>(req_type, resp_type);
    client_->Init(0, "devm_um_1");
    std::shared_ptr<common_req> req_ = std::make_shared<common_req>();
    req_->platform(1);
    
    for (int i  = 0; i < 100; i++) {
        if (0 == client_->WaitServiceOnline(0)) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::shared_ptr<update_status_resp> resp_ = std::make_shared<update_status_resp>();
    int iResult = client_->Request(req_, resp_, 5000);
    if (0 != iResult) {
        std::cout << "request failed." << std::endl;
        if (nullptr != client_) {
            client_->Deinit();
            client_ = nullptr;
        }
        return;
    }
    std::cout << "Resp is : " << std::endl;
    std::cout << "update_status : " << resp_->update_status() << std::endl;
    std::cout << "error code : " << resp_->error_code() << std::endl;
    std::cout << "error msg : " << resp_->error_msg() << std::endl;

    if (nullptr != client_) {
        client_->Deinit();
        client_ = nullptr;
    }
}

void testCaseStartUpdate(std::string path, int mode)
{
    std::shared_ptr<Client<start_update_req, start_update_resp>> client_;
    std::shared_ptr<start_update_reqPubSubType> req_type = std::make_shared<start_update_reqPubSubType>();
    std::shared_ptr<start_update_respPubSubType> resp_type = std::make_shared<start_update_respPubSubType>();
    client_ = std::make_shared<Client<start_update_req, start_update_resp>>(req_type, resp_type);
    client_->Init(0, "devm_um_2");
    std::shared_ptr<start_update_req> req_ = std::make_shared<start_update_req>();
    req_->start_with_precheck(true);
    req_->ecu_mode(mode);
    req_->package_path(path);
    
    for (int i  = 0; i < 100; i++) {
        if (0 == client_->WaitServiceOnline(0)) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::shared_ptr<start_update_resp> resp_ = std::make_shared<start_update_resp>();
    int iResult = client_->Request(req_, resp_, 5000);
    if (0 != iResult) {
        std::cout << "request failed." << std::endl;
        if (nullptr != client_) {
            client_->Deinit();
            client_ = nullptr;
        }
        return;
    }
    std::cout << "Resp is : " << std::endl;
    std::cout << "error code : " << resp_->error_code() << std::endl;
    std::cout << "error msg : " << resp_->error_msg() << std::endl;

    if (nullptr != client_) {
        client_->Deinit();
        client_ = nullptr;
    }
}

void testCaseProgress()
{
    std::shared_ptr<Client<common_req, progress_resp>> client_;
    std::shared_ptr<common_reqPubSubType> req_type = std::make_shared<common_reqPubSubType>();
    std::shared_ptr<progress_respPubSubType> resp_type = std::make_shared<progress_respPubSubType>();
    client_ = std::make_shared<Client<common_req, progress_resp>>(req_type, resp_type);
    client_->Init(0, "devm_um_3");
    std::shared_ptr<common_req> req_ = std::make_shared<common_req>();
    req_->platform(1);
    
    for (int i  = 0; i < 100; i++) {
        if (0 == client_->WaitServiceOnline(0)) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::shared_ptr<progress_resp> resp_ = std::make_shared<progress_resp>();
    int iResult = client_->Request(req_, resp_, 5000);
    if (0 != iResult) {
        std::cout << "request failed." << std::endl;
        if (nullptr != client_) {
            client_->Deinit();
            client_ = nullptr;
        }
        return;
    }
    std::cout << "Resp is : " << std::endl;
    std::cout << "progress : " << resp_->progress() << std::endl;
    std::cout << "error code : " << resp_->error_code() << std::endl;
    std::cout << "error msg : " << resp_->error_msg() << std::endl;

    if (nullptr != client_) {
        client_->Deinit();
        client_ = nullptr;
    }
}

void testCasePreCheck()
{
    std::shared_ptr<Client<common_req, precheck_resp>> client_;
    std::shared_ptr<common_reqPubSubType> req_type = std::make_shared<common_reqPubSubType>();
    std::shared_ptr<precheck_respPubSubType> resp_type = std::make_shared<precheck_respPubSubType>();
    client_ = std::make_shared<Client<common_req, precheck_resp>>(req_type, resp_type);
    client_->Init(0, "devm_um_4");
    std::shared_ptr<common_req> req_ = std::make_shared<common_req>();
    req_->platform(1);
    
    for (int i  = 0; i < 100; i++) {
        if (0 == client_->WaitServiceOnline(0)) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::shared_ptr<precheck_resp> resp_ = std::make_shared<precheck_resp>();
    int iResult = client_->Request(req_, resp_, 5000);
    if (0 != iResult) {
        std::cout << "request failed." << std::endl;
        if (nullptr != client_) {
            client_->Deinit();
            client_ = nullptr;
        }
        return;
    }
    std::cout << "Resp is : " << std::endl;
    std::cout << "space : " << resp_->space() << std::endl;
    std::cout << "speed : " << resp_->speed() << std::endl;
    std::cout << "gear : " << resp_->gear() << std::endl;
    std::cout << "error code : " << resp_->error_code() << std::endl;
    std::cout << "error msg : " << resp_->error_msg() << std::endl;

    if (nullptr != client_) {
        client_->Deinit();
        client_ = nullptr;
    }
}

void testCaseGetVersion()
{
    std::shared_ptr<Client<common_req, get_version_resp>> client_;
    std::shared_ptr<common_reqPubSubType> req_type = std::make_shared<common_reqPubSubType>();
    std::shared_ptr<get_version_respPubSubType> resp_type = std::make_shared<get_version_respPubSubType>();
    client_ = std::make_shared<Client<common_req, get_version_resp>>(req_type, resp_type);
    client_->Init(0, "devm_um_5");
    std::shared_ptr<common_req> req_ = std::make_shared<common_req>();
    req_->platform(1);
    
    for (int i  = 0; i < 100; i++) {
        if (0 == client_->WaitServiceOnline(0)) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::shared_ptr<get_version_resp> resp_ = std::make_shared<get_version_resp>();
    int iResult = client_->Request(req_, resp_, 5000);
    if (0 != iResult) {
        std::cout << "request failed." << std::endl;
        if (nullptr != client_) {
            client_->Deinit();
            client_ = nullptr;
        }
        return;
    }
    std::cout << "Resp is : " << std::endl;
    std::cout << "major version : " << resp_->major_version() << std::endl;
    std::cout << "soc version : " << resp_->soc_version() << std::endl;
    std::cout << "dsv version : " << resp_->dsv_version() << std::endl;
    std::cout << "mcu version : " << resp_->mcu_version() << std::endl;
    // 遍历并输出map的内容
    for(const auto& pair : resp_->sensor_version()) {
        std::cout << "Key: " << pair.first << ", Value: " << pair.second << std::endl;
    }
    std::cout << "error code : " << resp_->error_code() << std::endl;
    std::cout << "error msg : " << resp_->error_msg() << std::endl;

    if (nullptr != client_) {
        client_->Deinit();
        client_ = nullptr;
    }
}

void testCaseStartFinish()
{
    std::shared_ptr<Client<common_req, start_finish_resp>> client_;
    std::shared_ptr<common_reqPubSubType> req_type = std::make_shared<common_reqPubSubType>();
    std::shared_ptr<start_finish_respPubSubType> resp_type = std::make_shared<start_finish_respPubSubType>();
    client_ = std::make_shared<Client<common_req, start_finish_resp>>(req_type, resp_type);
    client_->Init(0, "devm_um_6");
    std::shared_ptr<common_req> req_ = std::make_shared<common_req>();
    req_->platform(1);
    
    for (int i  = 0; i < 100; i++) {
        if (0 == client_->WaitServiceOnline(0)) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::shared_ptr<start_finish_resp> resp_ = std::make_shared<start_finish_resp>();
    int iResult = client_->Request(req_, resp_, 5000);
    if (0 != iResult) {
        std::cout << "request failed." << std::endl;
        if (nullptr != client_) {
            client_->Deinit();
            client_ = nullptr;
        }
        return;
    }
    std::cout << "Resp is : " << std::endl;
    std::cout << "error code : " << resp_->error_code() << std::endl;
    std::cout << "error msg : " << resp_->error_msg() << std::endl;

    if (nullptr != client_) {
        client_->Deinit();
        client_ = nullptr;
    }
}

int main(int argc, char ** argv) {

    if (argc < 3)
    {
        std::cout << "error argv nums ." << std::endl;
        return 0;
    }
    
    std::string package_path = argv[1];
    std::string mode = argv[2];
    int ecu_mode = std::stoi(mode);

    int test_case;
    std::cout << "Enter the test case number (1-5): " << std::endl;
    std::cout << "number 1  -->  testCaseUpdateStatus " << std::endl;
    std::cout << "number 2  -->  testCaseStartUpdate " << std::endl;
    std::cout << "number 3  -->  testCaseProgress " << std::endl;
    std::cout << "number 4  -->  testCasePreCheck " << std::endl;
    std::cout << "number 5  -->  testCaseGetVersion " << std::endl;
    std::cout << "number 6  -->  testCaseStartFinish " << std::endl;


    std::cin >> test_case;

    switch (test_case) {
    case 1:
        std::thread(testCaseUpdateStatus).join();
        break;
    case 2:
        std::thread(testCaseStartUpdate, package_path, ecu_mode).join();
        break;
    case 3:
        std::thread(testCaseProgress).join();
        break;
    case 4:
        std::thread(testCasePreCheck).join();
        break;
    case 5:
        std::thread(testCaseGetVersion).join();
        break;
    case 6:
        std::thread(testCaseStartFinish).join();
        break;
    default:
        std::cout << "Invalid test case number\n";
        break;
    }
    return 0;
}
