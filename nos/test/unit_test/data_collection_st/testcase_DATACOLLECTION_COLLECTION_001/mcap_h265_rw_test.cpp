/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
*
* @File: all_file_merge_test.cpp
* @Date: 2023/11/29
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
#include "processor/include/impl/mcap_h265_rw.h"
#include "processor/include/impl/desense_manager.h"
#include "utils/include/path_utils.h"
#include "gtest/gtest.h"

using namespace hozon::netaos::dc;

class McapH265RWTest:public ::testing::Test {

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


YAML::Node getConfigureForDesen() {
    std::string cfg = R"(
    desensitization:
      type: DESENSE
      lifecycle: lifeCycleOnce
      configuration: []
)";
    YAML::Node node=YAML::Load(cfg);
    return node;
}

YAML::Node getConfigureForDesenManager() {
    std::string cfg = R"(
    desenseManager:
      type: DESENSEMANAGER
      lifecycle: lifeCycleOnce
      configuration:
      -  outputFolderPath: /opt/usr/col/bag/original/desense/
         enable: true
         delayMs: 0
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

TEST_F(McapH265RWTest, testcase_DATA_COLLECTION_MCAP_H265_RW_001) {
    DesenseManager itemManager;
    EXPECT_EQ(itemManager.getStatus(),TaskStatus::INITIAL);
    YAML::Node yamlConfigNode = getConfigureForDesenManager();
    for (auto node : yamlConfigNode["desenseManager"]["configuration"]) {
        itemManager.configure("default", node);
    }
    itemManager.active();

    McapH265RW item;
    EXPECT_EQ(item.getStatus(),TaskStatus::INITIAL);
    DataTrans inTran;
    std::vector<std::string> videoTopicMcapFileVec;
    PathUtils::getFiles("/opt/usr/col/bag/videorec/", videoTopicMcapFileVec, true);
    inTran.pathsList[videoTopicMcapFiles].insert(videoTopicMcapFileVec.begin(), videoTopicMcapFileVec.end());
    item.configure("default", inTran);
    EXPECT_EQ(item.getStatus(),TaskStatus::CONFIGURED);
    item.active();
    EXPECT_EQ(item.getStatus(),TaskStatus::FINISHED);

    itemManager.deactive();
    EXPECT_EQ(itemManager.getStatus(),TaskStatus::FINISHED);
    
}

int main(int argc, char* argv[]){
    hozon::netaos::log::InitLogging("DCTEST", "NETAOS DC", hozon::netaos::log::LogLevel::kTrace,
                                    hozon::netaos::log::HZ_LOG2CONSOLE , "/opt/usr/log/soc_log/", 10, (20*1024*1024),true);
    testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}
