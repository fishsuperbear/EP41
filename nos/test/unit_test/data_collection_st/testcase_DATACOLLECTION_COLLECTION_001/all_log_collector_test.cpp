/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: all_log_collector_test.cpp
 * @Date: 2023/11/28
 * @Author: cheng
 * @Desc: --
 */

#include <iostream>
#include <filesystem>

#include "log/include/logging.h"
#include "collection/include/impl/all_log_collector.h"
#include "utils/include/path_utils.h"
#include "gtest/gtest.h"

using namespace hozon::netaos::dc;

class AllLogCollectorTest:public ::testing::Test {

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
allLogCollectorLargeAndSlow:
  type: ALL_LARGE_LOG_COLLECTOR
  priority: 2
  lifecycle: lifeCycleOnce
  configuration:
    - type: all_log
      folderList:
        - path: /opt/usr/col/fm
          sizeMb: 10
        - path: /opt/usr/log/soc_log
          sizeMb: 1500
        - path: /opt/usr/log/system_monitor_log
          sizeMb: 15
        - path: /opt/usr/log/mcu_log
          sizeMb: 100
        - path: /opt/usr/log/ota_log
          sizeMb: 100
        - path: /opt/usr/log/svp_log
          sizeMb: 100
        - path: /opt/usr/mcu_adas
          sizeMb: 200
        - path: /opt/usr/col/runinfo
          sizeMb: 1
)";
//    std::string newCfg = cfg.replace(cfg.find("$"),2,bagPath);
    YAML::Node node=YAML::Load(cfg);
    return node;
}



TEST_F(AllLogCollectorTest, testcase_DATA_COLLECTION_BAG_REC_001) {

    AllLogCollector item;
    std::filesystem::path execPath = std::filesystem::current_path(); // 获取当前工作路径
    YAML::Node yamlConfigNode = getConfigure();
    for (auto node : yamlConfigNode["allLogCollectorLargeAndSlow"]["configuration"]) {
        if (node["type"]) {
            item.configure(node["type"].as<std::string>(), node);
        } else {
            item.configure("default", node);
        }
    }
    EXPECT_EQ(item.getStatus(),TaskStatus::CONFIGURED);
    item.active();
    EXPECT_EQ(item.getStatus(),TaskStatus::FINISHED);
    item.deactive();
    EXPECT_EQ(item.getStatus(),TaskStatus::FINISHED);
    struct DataTrans dataStruct;
    item.getTaskResult("default", dataStruct);
    EXPECT_GT(dataStruct.pathsList[hzLogFiles].size(), 0);

    for (const auto& type2PathList : dataStruct.pathsList) {
        DC_CLIENT_LOG_DEBUG<<"=====path type: "<<type2PathList.first;
        for (auto &path : type2PathList.second) {
            if (PathUtils::isFileExist(path)) {
                DC_CLIENT_LOG_DEBUG<<"\t\tf: "<<path;
            } else if (PathUtils::isDirExist(path)) {
                DC_CLIENT_LOG_DEBUG<<"\t\tfolder: "<<path;
            }
        }
    }
}

int main(int argc, char* argv[]){
    hozon::netaos::log::InitLogging("DCTEST", "NETAOS DC", hozon::netaos::log::LogLevel::kTrace,
                                    hozon::netaos::log::HZ_LOG2CONSOLE , "/opt/usr/log/soc_log/", 10, (20*1024*1024),true);
    testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}
