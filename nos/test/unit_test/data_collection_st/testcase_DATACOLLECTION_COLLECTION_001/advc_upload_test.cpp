/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
*
* @File: advc_upload_test.cpp
* @Date: 2023/11/20
* @Author: cheng
* @Desc: --
*/
//#define private public
//#define protected public
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <filesystem>

#include "destination/include/advc_upload.h"
#include "include/magic_enum.hpp"
#include "log/include/logging.h"
#include "utils/include/path_utils.h"
#include "gtest/gtest.h"

using namespace hozon::netaos::dc;


class AdvcUploadTest:public ::testing::Test {

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

YAML::Node getConfigure(std::string upload_type) {
    std::string cfg = R"(
uploadFiles:
    version: 1
    type: "default"
    lifecycle: lifeCycleOnce
    configuration:
    - type: default
      uploadPathFormat: "%P_%t_%s"
      protocolId: "PROTO"
      retryCount: 5
      retryInterval: 1000
      uploadType: "$t"
      checkPoint: ""
      enableEncrypt: false
)";
    std::string newCfg = cfg.replace(cfg.find("$"),2,upload_type);
    YAML::Node node=YAML::Load(newCfg);
    return node;
}
void IEXPECT_EQ(TaskStatus expected, TaskStatus target, std::string description) {
    if (expected!=target) {
        std::cout<<"++++++++++ERROR:"<<description<<":NOK:"<<magic_enum::enum_name(expected)<<"!="<<magic_enum::enum_name(target)<<std::endl;
    }
    std::cout<<"++++++++++"<<description<<":OK"<<std::endl;
}


TEST_F(AdvcUploadTest, testcase_DATA_COLLECTION_ADVC_UPLOAD_001) {

    AdvcUploadTask item;
    ThreadPoolFlex tp(2,2);
    item.setThreadPool(&tp);

    std::filesystem::path execPath = std::filesystem::current_path(); // 获取当前工作路径
    auto folderPath = execPath.parent_path().string()+"/dc_upload";
    auto filePath = folderPath+"/EP41_ORIN_log_2023upload_test.log";
    YAML::Node yamlConfigNode = getConfigure("LOG");
    PathUtils::createFoldersIfNotExists(folderPath);
    std::ofstream outfile(filePath);
    outfile << "Hello, world! just for upload small files" << std::endl;
    outfile.close();
    DataTrans inputNode;
    inputNode.pathsList[faultManagerFiles].insert(filePath);
    inputNode.dataType = DataTransType::fileAndFolder;

    for (auto node : yamlConfigNode["uploadFiles"]["configuration"]) {
        if (node["type"]) {
            item.configure(node["type"].as<std::string>(), node);
        } else {
            item.configure("default", node);
        }
    }
    item.configure("default", inputNode);

    EXPECT_EQ(item.getStatus(),TaskStatus::CONFIGURED);
    item.active();
    EXPECT_EQ(item.getStatus(),TaskStatus::RUNNING);
    std::this_thread::sleep_for(std::chrono::seconds(15));
    item.deactive();
    EXPECT_EQ(item.getStatus(),TaskStatus::FINISHED);
    tp.stop();
    EXPECT_FALSE(std::filesystem::is_empty(folderPath));
    PathUtils::removeFilesInFolder(folderPath);
}

int main(int argc, char* argv[]){
    hozon::netaos::log::InitLogging("DCTEST", "NETAOS DC", hozon::netaos::log::LogLevel::kError,
                                    hozon::netaos::log::HZ_LOG2CONSOLE , "/opt/usr/log/soc_log/", 10, (20*1024*1024),true);
    testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}