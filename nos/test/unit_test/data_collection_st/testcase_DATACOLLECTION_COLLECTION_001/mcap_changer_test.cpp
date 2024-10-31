/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
*
* @File: mcap_changer_test.cpp
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
#include "processor/include/impl/mcap_changer.h"
#include "utils/include/path_utils.h"
#include "gtest/gtest.h"

using namespace hozon::netaos::dc;

class McapChangerTest:public ::testing::Test {

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
    mcap_filter:
      type: MCAPDEAL
      lifecycle: lifeCycleOnce
      configuration:
      -   method: filter
          whiteTopicVec: []
          blackTopicVec: []
          outputPath: /opt/usr/col/bag/mcap/filter/
    mcap_merge:
      type: MCAPDEAL
      lifecycle: lifeCycleOnce
      configuration:
      -   method: filter
          attachmentFilePathVec: [/app/version.json]
          outputPath: /opt/usr/col/bag/mcap/merge/
    mcap_split:
      type: MCAPDEAL
      lifecycle: lifeCycleOnce
      configuration:
      -   method: filter
          attachmentFilePathVec: [/app/version.json]
          outputPath: /opt/usr/col/bag/mcap/split/
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


TEST_F(McapChangerTest, testcase_DATA_COLLECTION_MCAP_FILTER_001) {
    McapChanger item;
    EXPECT_EQ(item.getStatus(),TaskStatus::INITIAL);
    YAML::Node yamlConfigNode = getConfigure();
    for (auto node : yamlConfigNode["mcap_filter"]["configuration"]) {
        item.configure("default", node);
    }
    EXPECT_EQ(item.getStatus(),TaskStatus::CONFIGURED);
    DataTrans inTran;
    std::vector<std::string> commonTopicMcapFileVec;
    PathUtils::getFiles("/opt/usr/col/bag/commonrec/", commonTopicMcapFileVec, true);
    if (!commonTopicMcapFileVec.empty()) {
        inTran.pathsList[commonTopicMcapFiles].insert(commonTopicMcapFileVec[0]);
        item.configure("default", inTran);
        item.active();
        EXPECT_EQ(item.getStatus(),TaskStatus::FINISHED);
        DataTrans outTran;
        EXPECT_TRUE(item.getTaskResult("default", outTran));
    }
}

TEST_F(McapChangerTest, testcase_DATA_COLLECTION_MCAP_MERGE_001) {
    McapChanger item;
    EXPECT_EQ(item.getStatus(),TaskStatus::INITIAL);
    YAML::Node yamlConfigNode = getConfigure();
    for (auto node : yamlConfigNode["mcap_merge"]["configuration"]) {
        item.configure("default", node);
    }
    EXPECT_EQ(item.getStatus(),TaskStatus::CONFIGURED);
    DataTrans inTran;
    std::vector<std::string> commonTopicMcapFileVec;
    PathUtils::getFiles("/opt/usr/col/bag/commonrec/", commonTopicMcapFileVec, true);
    if (commonTopicMcapFileVec.size() > 1) {
        inTran.pathsList[commonTopicMcapFiles].insert(commonTopicMcapFileVec.begin(), commonTopicMcapFileVec.begin() + 1);
        item.configure("default", inTran);
        item.active();
        EXPECT_EQ(item.getStatus(),TaskStatus::FINISHED);
        DataTrans outTran;
        EXPECT_TRUE(item.getTaskResult("default", outTran));
    }
}

TEST_F(McapChangerTest, testcase_DATA_COLLECTION_MCAP_SPLIT_001) {
    McapChanger item;
    EXPECT_EQ(item.getStatus(),TaskStatus::INITIAL);
    YAML::Node yamlConfigNode = getConfigure();
    for (auto node : yamlConfigNode["mcap_split"]["configuration"]) {
        item.configure("default", node);
    }
    EXPECT_EQ(item.getStatus(),TaskStatus::CONFIGURED);
    DataTrans inTran;
    std::vector<std::string> commonTopicMcapFileVec;
    PathUtils::getFiles("/opt/usr/col/bag/mcap/merge/", commonTopicMcapFileVec, true);
    if (!commonTopicMcapFileVec.empty()) {
        inTran.pathsList[commonTopicMcapFiles].insert(commonTopicMcapFileVec[0]);
        item.configure("default", inTran);
        item.active();
        EXPECT_EQ(item.getStatus(),TaskStatus::FINISHED);
        DataTrans outTran;
        EXPECT_TRUE(item.getTaskResult("default", outTran));
    }
}

int main(int argc, char* argv[]){
    hozon::netaos::log::InitLogging("DCTEST", "NETAOS DC", hozon::netaos::log::LogLevel::kTrace,
                                    hozon::netaos::log::HZ_LOG2CONSOLE , "/opt/usr/log/soc_log/", 10, (20*1024*1024),true);
    testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}
