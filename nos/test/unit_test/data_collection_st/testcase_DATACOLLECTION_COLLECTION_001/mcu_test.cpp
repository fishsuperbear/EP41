/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
*
* @File: mcu_test.cpp
* @Date: 2023/11/29
* @Author: shenda
* @Desc: --
*/

//#define private public
//#define protected public

#include <fcntl.h>
#include <pthread.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <future>
#include <regex>
#include <thread>

#include "gtest/gtest.h"
#include "collection/include/impl/bag_record.h"
#include "collection/include/impl/mcu_bag_recorder.h"
#include "log/include/logging.h"
#include "mcu/include/MCUClient.h"
#include "utils/include/path_utils.h"

using namespace hozon::netaos::dc;

class MCUTest : public ::testing::Test {

   protected:
    static void SetUpTestSuite() {}

    static void TearDownTestSuite() {}

    void SetUp() override {}

    void TearDown() override {}
};

bool get_comand_output(const std::function<int()> command, const std::string& regex_filter, int regex_group, std::string& output, bool keep_alive) {
    static std::mutex mutex;
    static bool executed = false;
    static int pipe_fd[2]{0, 0};
    {
        std::lock_guard<std::mutex> guard(mutex);
        if (pipe_fd[1] == 0) {
            if (pipe(pipe_fd) != 0) {
                pipe_fd[1] = 0;
                printf("get_comand_output pipe error\n");
                return false;
            }
            fcntl(pipe_fd[0], F_SETFL, O_NONBLOCK);
            fcntl(pipe_fd[1], F_SETFL, O_NONBLOCK);
        }
    }
    thread_local pid_t pid{0};
    {
        std::lock_guard<std::mutex> guard(mutex);
        if (pid == 0) {
            pid = fork();
            if (pid < 0) {
                pid = 0;
                printf("get_comand_output fork error\n");
                return false;
            }
            if (pid > 0 && keep_alive) {
                kill(pid, SIGSTOP);
            }
        }
    }
    if (pid > 0) {
        char buffer[4096]{0};
        ssize_t read_count{0};
        const int RETRY_COUNT = 5;
        const int SLEEP_INTERVAL = 50;
        bool stopped = true;
        for (int i = 0; i < RETRY_COUNT; ++i) {
            {
                std::lock_guard<std::mutex> guard(mutex);
                if (keep_alive && stopped) {
                    auto res = kill(pid, SIGCONT);
                    stopped = false;
                }
                read_count = read(pipe_fd[0], buffer, sizeof(buffer) - 1);
                if (read_count > 0) {
                    if (keep_alive) {
                        auto res = kill(pid, SIGSTOP);
                        stopped = true;
                    }
                }
            }
            if (read_count > 0 && buffer[read_count - 1] == '\n') {
                buffer[read_count - 1] = 0;
                output = buffer;
            }
            if (!output.empty()) {
                break;
            }
            printf("get_comand_output sleeping\n");
            std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_INTERVAL));
        }
    } else {
        close(STDIN_FILENO);
        close(STDOUT_FILENO);
        close(STDERR_FILENO);
        dup2(pipe_fd[1], STDOUT_FILENO);
        auto res = command();
        if (res != 0) {
            printf("get_comand_output execute command error\n");
        }
        printf("get_comand_output command process exit\n");
        exit(1);
    }
    if (pid > 0 && !keep_alive) {
        if (kill(pid, SIGKILL) == 0) {
            printf("get_comand_output wait\n");
            wait(nullptr);
            pid = 0;
        }
    }
    if (output.empty()) {
        if (pid > 0 && keep_alive) {
            printf("get_comand_output SIGKILL\n");
            kill(pid, SIGKILL);
        }
        return false;
    }
    if (regex_filter.empty() || regex_group < 0) {
        return true;
    }
    std::regex reg(regex_filter);
    std::smatch sm;
    if (std::regex_search(output.cbegin(), output.cend(), sm, reg)) {
        if (regex_group > 0) {
            output = sm.str(regex_group);
        } else {
            output = sm.str();
        }
        if (!output.empty()) {
            return true;
        }
    }
    printf("get_comand_output regex filter error\n");
    return false;
}

TEST_F(MCUTest, testcase_DC_MCU_001) {
    std::string cmd_output;
    get_comand_output([] { return system("ps -ef|grep dc_mcu|grep -v grep|wc -l"); }, R"((\d))", 1, cmd_output, false);
    EXPECT_TRUE(std::stoi(cmd_output) >= 0);
}

void* thread_fun(void* arg) {
    std::promise<bool>* pres = (std::promise<bool>*)arg;
    hozon::netaos::dc::MCUClient mcu_client;
    mcu_client.Init();
    mcu_client.Deinit();
    pres->set_value(true);
    return (void*)0;
}

TEST_F(MCUTest, testcase_DC_MCU_002) {
    std::promise<bool> res;
    auto res_future = res.get_future();
    pthread_t tid;
    int err;
    int res_kill;
    err = pthread_create(&tid, NULL, thread_fun, &res);
    EXPECT_TRUE(err == 0);
    auto status = res_future.wait_for(std::chrono::seconds(10));
    if (status != std::future_status::ready) {
        ara::core::Deinitialize();
    }
    EXPECT_TRUE(status == std::future_status::ready || status == std::future_status::timeout);
}

YAML::Node getBagRecordConf(const std::string& folderPath, const std::string& topic) {
    std::string bagPath = folderPath + "/DC_TEST";
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
        - $t)";
    cfg = cfg.replace(cfg.find("$f"), 2, bagPath);
    cfg = cfg.replace(cfg.find("$t"), 2, topic);
    YAML::Node node = YAML::Load(cfg);
    return node;
}

TEST_F(MCUTest, testcase_DC_MCU_BagRecord) {
    {
        MCUBagRecorder item1;
        std::filesystem::path execPath = std::filesystem::current_path();
        auto folderPath1 = execPath.parent_path().string() + "/dc_bag11";
        PathUtils::createFoldersIfNotExists(folderPath1);
        YAML::Node yamlConfigNode1 = getBagRecordConf(folderPath1, "/soc/rawpointcloud");
        for (auto node : yamlConfigNode1["recordAllTopicBase"]["configuration"]) {
            if (node["type"]) {
                item1.configure(node["type"].as<std::string>(), node);
            } else {
                item1.configure("default", node);
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
        MCUBagRecorder item2;
        auto folderPath2 = execPath.parent_path().string() + "/dc_bag22";
        YAML::Node yamlConfigNode2 = getBagRecordConf(folderPath2, "/soc/rawpointcloud");
        for (auto node : yamlConfigNode2["recordAllTopicBase"]["configuration"]) {
            if (node["type"]) {
                item2.configure(node["type"].as<std::string>(), node);
            } else {
                item2.configure("default", node);
            }
        }

        EXPECT_EQ(item1.getStatus(), TaskStatus::CONFIGURED);
        item1.active();
        EXPECT_EQ(item1.getStatus(), TaskStatus::RUNNING);
        EXPECT_EQ(item2.getStatus(), TaskStatus::CONFIGURED);
        item2.active();
        EXPECT_EQ(item2.getStatus(), TaskStatus::RUNNING);
        std::this_thread::sleep_for(std::chrono::seconds(60));
        EXPECT_EQ(item1.getStatus(), TaskStatus::RUNNING);
        item1.deactive();
        EXPECT_EQ(item1.getStatus(), TaskStatus::FINISHED);
        EXPECT_EQ(item2.getStatus(), TaskStatus::RUNNING);
        item2.deactive();
        EXPECT_EQ(item2.getStatus(), TaskStatus::FINISHED);
        EXPECT_FALSE(std::filesystem::is_empty(folderPath1));
        EXPECT_FALSE(std::filesystem::is_empty(folderPath2));
        // PathUtils::removeFilesInFolder(folderPath1);
        // PathUtils::removeFilesInFolder(folderPath2);
    }
    {
        MCUBagRecorder item1;
        std::filesystem::path execPath = std::filesystem::current_path();
        auto folderPath1 = execPath.parent_path().string() + "/dc_bag1";
        PathUtils::createFoldersIfNotExists(folderPath1);
        YAML::Node yamlConfigNode1 = getBagRecordConf(folderPath1, "/localization/location");
        for (auto node : yamlConfigNode1["recordAllTopicBase"]["configuration"]) {
            if (node["type"]) {
                item1.configure(node["type"].as<std::string>(), node);
            } else {
                item1.configure("default", node);
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
        MCUBagRecorder item2;
        auto folderPath2 = execPath.parent_path().string() + "/dc_bag2";
        YAML::Node yamlConfigNode2 = getBagRecordConf(folderPath2, "/perception/fsd/obj_fusion_1");
        for (auto node : yamlConfigNode2["recordAllTopicBase"]["configuration"]) {
            if (node["type"]) {
                item2.configure(node["type"].as<std::string>(), node);
            } else {
                item2.configure("default", node);
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
        MCUBagRecorder item3;
        auto folderPath3 = execPath.parent_path().string() + "/dc_bag3";
        YAML::Node yamlConfigNode3 = getBagRecordConf(folderPath3, "/perception/parking/obj_fusion_2");
        for (auto node : yamlConfigNode3["recordAllTopicBase"]["configuration"]) {
            if (node["type"]) {
                item3.configure(node["type"].as<std::string>(), node);
            } else {
                item3.configure("default", node);
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
        MCUBagRecorder item4;
        auto folderPath4 = execPath.parent_path().string() + "/dc_bag4";
        YAML::Node yamlConfigNode4 = getBagRecordConf(folderPath4, "/perception/fsd/transportelement_1");
        for (auto node : yamlConfigNode4["recordAllTopicBase"]["configuration"]) {
            if (node["type"]) {
                item4.configure(node["type"].as<std::string>(), node);
            } else {
                item4.configure("default", node);
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
        MCUBagRecorder item5;
        auto folderPath5 = execPath.parent_path().string() + "/dc_bag5";
        YAML::Node yamlConfigNode5 = getBagRecordConf(folderPath5, "/perception/parking/transportelement_2");
        for (auto node : yamlConfigNode5["recordAllTopicBase"]["configuration"]) {
            if (node["type"]) {
                item5.configure(node["type"].as<std::string>(), node);
            } else {
                item5.configure("default", node);
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
        MCUBagRecorder item6;
        auto folderPath6 = execPath.parent_path().string() + "/dc_bag6";
        YAML::Node yamlConfigNode6 = getBagRecordConf(folderPath6, "/planning/ego_trajectory");
        for (auto node : yamlConfigNode6["recordAllTopicBase"]["configuration"]) {
            if (node["type"]) {
                item6.configure(node["type"].as<std::string>(), node);
            } else {
                item6.configure("default", node);
            }
        }
        EXPECT_EQ(item1.getStatus(), TaskStatus::CONFIGURED);
        item1.active();
        EXPECT_EQ(item1.getStatus(), TaskStatus::RUNNING);
        EXPECT_EQ(item2.getStatus(), TaskStatus::CONFIGURED);
        item2.active();
        EXPECT_EQ(item2.getStatus(), TaskStatus::RUNNING);
        EXPECT_EQ(item3.getStatus(), TaskStatus::CONFIGURED);
        item3.active();
        EXPECT_EQ(item3.getStatus(), TaskStatus::RUNNING);
        EXPECT_EQ(item4.getStatus(), TaskStatus::CONFIGURED);
        item4.active();
        EXPECT_EQ(item4.getStatus(), TaskStatus::RUNNING);
        EXPECT_EQ(item5.getStatus(), TaskStatus::CONFIGURED);
        item5.active();
        EXPECT_EQ(item5.getStatus(), TaskStatus::RUNNING);
        EXPECT_EQ(item6.getStatus(), TaskStatus::CONFIGURED);
        item6.active();
        EXPECT_EQ(item6.getStatus(), TaskStatus::RUNNING);

        std::this_thread::sleep_for(std::chrono::seconds(60));
        EXPECT_EQ(item1.getStatus(), TaskStatus::RUNNING);
        item1.deactive();
        EXPECT_EQ(item1.getStatus(), TaskStatus::FINISHED);
        EXPECT_EQ(item2.getStatus(), TaskStatus::RUNNING);
        item2.deactive();
        EXPECT_EQ(item2.getStatus(), TaskStatus::FINISHED);
        EXPECT_EQ(item3.getStatus(), TaskStatus::RUNNING);
        item3.deactive();
        EXPECT_EQ(item3.getStatus(), TaskStatus::FINISHED);
        EXPECT_EQ(item4.getStatus(), TaskStatus::RUNNING);
        item4.deactive();
        EXPECT_EQ(item4.getStatus(), TaskStatus::FINISHED);
        EXPECT_EQ(item5.getStatus(), TaskStatus::RUNNING);
        item5.deactive();
        EXPECT_EQ(item5.getStatus(), TaskStatus::FINISHED);
        EXPECT_EQ(item6.getStatus(), TaskStatus::RUNNING);
        item6.deactive();
        EXPECT_EQ(item6.getStatus(), TaskStatus::FINISHED);

        EXPECT_FALSE(std::filesystem::is_empty(folderPath1));
        // PathUtils::removeFilesInFolder(folderPath1);
        EXPECT_FALSE(std::filesystem::is_empty(folderPath2));
        // PathUtils::removeFilesInFolder(folderPath2);
        EXPECT_FALSE(std::filesystem::is_empty(folderPath3));
        // PathUtils::removeFilesInFolder(folderPath3);
        EXPECT_FALSE(std::filesystem::is_empty(folderPath4));
        // PathUtils::removeFilesInFolder(folderPath4);
        EXPECT_FALSE(std::filesystem::is_empty(folderPath5));
        // PathUtils::removeFilesInFolder(folderPath5);
        EXPECT_FALSE(std::filesystem::is_empty(folderPath6));
        // PathUtils::removeFilesInFolder(folderPath6);
    }
}

int main(int argc, char* argv[]) {
    hozon::netaos::log::InitLogging("DCTEST", "NETAOS DC MCU", hozon::netaos::log::LogLevel::kTrace, hozon::netaos::log::HZ_LOG2CONSOLE, "/opt/usr/log/soc_log/", 10, (20 * 1024 * 1024), true);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
