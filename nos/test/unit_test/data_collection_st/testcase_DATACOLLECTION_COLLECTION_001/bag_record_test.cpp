/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
*
* @File: bag_record_test.cpp
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

#include "log/include/logging.h"
#include "collection/include/impl/bag_record.h"
#include "utils/include/path_utils.h"
#include "gtest/gtest.h"

using namespace hozon::netaos::dc;

class BagRecordTest:public ::testing::Test {

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


YAML::Node getConfigure(std::string &folderPath) {
    std::string bagPath = folderPath+"/DC_TEST" ;
    std::string cfg = R"(
   recordAllTopicBase:
     lifecycle: lifeCycleOnce
     configuration:
       - type: storageOption
         version: 1
         max_bagfile_duration: 1
         max_files: 17
         output_file_name: "$f"
         record_all: false
         topics:
           - /soc/chassis
           - /soc/imuinsinfo
           - /soc/gnssinfo
           - /soc/radarcorner1
           - /soc/radarcorner2
           - /soc/radarcorner3
           - /soc/radarcorner4
           - /soc/radarfront
           - /soc/ussrawdata
           - /soc/statemachine
           - /soc/mcu2ego
           - /soc/chassis_ota_method
           - /soc/apa2mcu_chassis
           - /perception/fsd/obj_fusion_1
           - /perception/fsd/freespace_1
           - /perception/fsd/transportelement_1
           - /perception/parking/obj_fusion_2
           - /perception/parking/freespace_2
           - /perception/parking/transportelement_2
           - /perception/parking/parkinglot_2
           - /perception/parking/state_machine
           - /perception/parking/slam_location
           - /planning/ego_trajectory
           - /planning/routing
           - /planning/prediction
           - /planning/ego_planning dec
           - /localization/fusionmap
           - /localization/location
           - /localization/deadreckoning
           - /localization/location_node
           - /soc/encoded_camera_0
           - /soc/encoded_camera_1
           - /soc/encoded_camera_3
           - /soc/encoded_camera_4
           - /soc/encoded_camera_5
           - /soc/encoded_camera_6
           - /soc/encoded_camera_7
           - /soc/encoded_camera_8
)";
    std::string newCfg = cfg.replace(cfg.find("$"),2,bagPath);
    YAML::Node node=YAML::Load(newCfg);
    return node;
}
void IEXPECT_EQ(TaskStatus expected, TaskStatus target, std::string description) {
    if (expected!=target) {
        std::cout<<"ERROR!!!:"<<description<<":NOK"<<std::endl;
    }
    std::cout<<description<<":OK"<<std::endl;
}


TEST_F(BagRecordTest, testcase_DATA_COLLECTION_BAG_REC_001) {

    BagRecorder item;
    std::filesystem::path execPath = std::filesystem::current_path(); // 获取当前工作路径
    auto folderPath = execPath.parent_path().string()+"/dc_bag";
    PathUtils::createFoldersIfNotExists(folderPath);
    YAML::Node yamlConfigNode = getConfigure(folderPath);
    for (auto node : yamlConfigNode["recordAllTopicBase"]["configuration"]) {
        if (node["type"]) {
            item.configure(node["type"].as<std::string>(), node);
        } else {
            item.configure("default", node);
        }
    }
    EXPECT_EQ(item.getStatus(),TaskStatus::CONFIGURED);
    item.active();
    EXPECT_EQ(item.getStatus(),TaskStatus::RUNNING);
    std::this_thread::sleep_for(std::chrono::seconds(5));
    EXPECT_EQ(item.getStatus(),TaskStatus::RUNNING);
    item.deactive();
    EXPECT_EQ(item.getStatus(),TaskStatus::FINISHED);
    EXPECT_FALSE(std::filesystem::is_empty(folderPath));
    PathUtils::removeFilesInFolder(folderPath);
    EXPECT_TRUE(std::filesystem::is_empty(folderPath));
}

int main(int argc, char* argv[]){
    hozon::netaos::log::InitLogging("DCTEST", "NETAOS DC", hozon::netaos::log::LogLevel::kTrace,
                                    hozon::netaos::log::HZ_LOG2CONSOLE , "/opt/usr/log/soc_log/", 10, (20*1024*1024),true);
    testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}
