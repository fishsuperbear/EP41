/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
*
* @File: add_data_test.cpp
* @Date: 2023/11/28
* @Author: kun
* @Desc: --
*/
//#define private public
//#define protected public
#include <iostream>
#include <thread>
#include <chrono>
#include <filesystem>

#include "log/include/logging.h"
#include "processor/include/impl/add_data.h"
#include "utils/include/path_utils.h"
#include "gtest/gtest.h"

using namespace hozon::netaos::dc;

class AddDataTest:public ::testing::Test {

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


YAML::Node getConfigure() {
    std::string cfg = R"(
    addData:
      type: ADDDATA
      lifecycle: lifeCycleOnce
      configuration:
      -  cmd:
           devm_version: nos devm upgrade version
         file:
           version: /app/version.json
         calibParamsVec:
         -  conf_calib_7v/7v_front_120_cam_params
         -  conf_calib_7v/7v_front_30_cam_params
         -  conf_calib_7v/7v_left_front_cam_params
         -  conf_calib_7v/7v_left_rear_cam_params
         -  conf_calib_7v/7v_rear_cam_params
         -  conf_calib_7v/7v_right_front_cam_params
         -  conf_calib_7v/7v_right_rear_cam_params
         -  conf_calib_lidar/roof_lidar_params
)";
    YAML::Node node=YAML::Load(cfg);
    return node;
}
void IEXPECT_EQ(TaskStatus expected, TaskStatus target, std::string description) {
    if (expected!=target) {
        std::cout<<"ERROR!!!:"<<description<<":NOK"<<std::endl;
    }
    std::cout<<description<<":OK"<<std::endl;
}


TEST_F(AddDataTest, testcase_DATA_COLLECTION_ADD_DATA_001) {

    AddData item;
    YAML::Node yamlConfigNode = getConfigure();
    for (auto node : yamlConfigNode["addData"]["configuration"]) {
        item.configure("default", node);
    }
    EXPECT_EQ(item.getStatus(),TaskStatus::CONFIGURED);
    item.active();
    EXPECT_EQ(item.getStatus(),TaskStatus::FINISHED);
    DataTrans tran;
    std::string type;
    EXPECT_TRUE(item.getTaskResult(type, tran));
}

int main(int argc, char* argv[]){
    hozon::netaos::log::InitLogging("DCTEST", "NETAOS DC", hozon::netaos::log::LogLevel::kTrace,
                                    hozon::netaos::log::HZ_LOG2CONSOLE , "/opt/usr/log/soc_log/", 10, (20*1024*1024),true);
    testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}
