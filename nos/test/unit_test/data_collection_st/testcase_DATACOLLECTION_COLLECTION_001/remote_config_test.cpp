/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
*
* @File: remote_config_test.cpp
* @Date: 2023/11/20
* @Author: cheng
* @Desc: --
*/
//#define private public
//#define protected public
#include <iostream>
#include <thread>
#include <chrono>
#include <filesystem>
#include <csignal>
#include <csignal>
#include <thread>

#include "tsp_comm.h"
#include "sys/stat.h"
#include "unistd.h"

#include "log/include/logging.h"
#include "utils/include/path_utils.h"
#include "gtest/gtest.h"

using namespace hozon::netaos::dc;
using namespace hozon::netaos::https;

class RemoteConfigTest:public ::testing::Test {

   protected:
    static void SetUpTestSuite() {
        std::cout << "=== SetUpTestSuite ===" << std::endl;
    }

    static void TearDownTestSuite() {
        std::cout << "=== TearDownTestSuite ===" << std::endl;
    }

    void SetUp() override {
        //        system("export LD_LIBRARY_PATH=/app/lib:$LD_LIBRARY_PATH");
    }

    void TearDown() override {}

};



TEST_F(RemoteConfigTest, testcase_DATA_COLLECTION_TEMPLATE_001) {
    TspComm::GetInstance().Init();
    std::future<TspComm::TspResponse> ret_uuid = TspComm::GetInstance().RequestHdUuid();
    TspComm::TspResponse ret_request = ret_uuid.get();
    std::cout << "result_code:" << ret_request.result_code << " uuid:" << ret_request.response << std::endl;

    std::future<TspComm::TspResponse> ret_remotecfg = TspComm::GetInstance().RequestRemoteConfig();
    ret_request = ret_remotecfg.get();
    std::cout << "result_code:" << ret_request.result_code << " remoteconfig:" << ret_request.response << std::endl;

    std::future<TspComm::TspResponse> ret_uptoken = TspComm::GetInstance().RequestUploadToken();
    ret_request = ret_uptoken.get();
    std::cout << "result_code:" << ret_request.result_code << " uploadToken:" << ret_request.response << std::endl;
}

int main(int argc, char* argv[]){
    hozon::netaos::log::InitLogging("DCTEST", "NETAOS DC", hozon::netaos::log::LogLevel::kTrace,
                                    hozon::netaos::log::HZ_LOG2CONSOLE , "/opt/usr/log/soc_log/", 10, (20*1024*1024),true);
    testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}
