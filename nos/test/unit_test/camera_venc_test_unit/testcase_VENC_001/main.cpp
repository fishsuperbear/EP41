#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include "gtest/gtest.h"

#include "camera_venc_config.h"
#include "orin/yuv_nvstream_receiver_v2.h"

using namespace hozon::netaos::cameravenc;

bool g_write_to_file = false;
int8_t g_consumer_index = 0;
std::string g_conf_path = "/app/runtime_service/camera_venc/conf/camera_venc_conf.yaml";

YuvNvStreamReceiverV2 g_yuv_receiver;

class VencTest : public ::testing::TestWithParam<YuvNvStreamReceiverV2*> {
   public:
    std::atomic_uint8_t check_count_{0};

   protected:
    void SetUp() override { handle_ = GetParam(); }

    void TearDown() override {}

    bool LoadCfg() {
        auto cfg = CameraVencConfig::LoadConfig(g_conf_path);
        if (!cfg) {
            return false;
        }

        for (auto it : cfg->sensor_infos) {
            sensor_info_mapping_[it.yuv_topic] = it;

            // init file handle
            if (g_write_to_file) {
                size_t delm = it.yuv_topic.rfind("/");
                std::string file_name = it.yuv_topic.substr(delm + 1, it.yuv_topic.size() - delm - 1);
                file_name += ".265";
                file_map_[it.yuv_topic] = std::make_unique<std::ofstream>(file_name, std::ios::binary | std::ios::out);
            }
        }

        for (auto it = sensor_info_mapping_.begin(); it != sensor_info_mapping_.end();) {
            bool set = false;
            for (auto id : cfg->sensor_ids) {
                if (it->second.sensor_id == id) {
                    set = true;
                    break;
                }
            }

            if (!set) {
                it = sensor_info_mapping_.erase(it);
            } else {
                ++it;
            }
        }
        return true;
    }

    void OnH265Data(std::string topic, hozon::netaos::desay::Multicast_EncodedImage& encoded_image) {
        check_cb_(encoded_image.data.size());
        check_count_++;
    }

    bool Init() {
        handle_->SetSensorInfos(sensor_info_mapping_);
        for (auto& it : sensor_info_mapping_) {
            hozon::netaos::desay::EncConsumerCbs cbs;
            cbs.encoded_image_cb = std::bind(&VencTest::OnH265Data, this, it.first, std::placeholders::_1);
            handle_->SetCallbacks(it.first, cbs);
        }
        return handle_->Init();
    }

    void SetCheckImgCb(std::function<void(uint32_t)> cb) { check_cb_ = cb; }

   protected:
    YuvNvStreamReceiverV2* handle_;
    SensorInfoMap sensor_info_mapping_;
    std::unordered_map<std::string, std::unique_ptr<std::ofstream>> file_map_;
    std::function<void(uint32_t)> check_cb_;
};

bool TerminateProcessByName(const char* processName) {
    char command[256];
    snprintf(command, sizeof(command), "killall -2 %s", processName);

    int result = system(command);
    if (result == -1) {
        std::cerr << "Failed to execute command." << std::endl;
        return false;
    }

    return true;
}

void StartProcess(const std::string& cmd) {
    std::system(cmd.data());
}

TEST_P(VencTest, NvstreamConnect) {
    EXPECT_TRUE(handle_ != nullptr);
    EXPECT_TRUE(LoadCfg());
    std::string cmd("/app/runtime_service/nvs_producer/bin/nvs_producer");
    std::future<void> res = std::async(std::launch::async, StartProcess, cmd);
    EXPECT_TRUE(Init());
    std::cout << "connect success.\n";
    auto ret = TerminateProcessByName("nvs_producer");
    EXPECT_TRUE(ret);
    res.get();
    EXPECT_EQ(0, 0);
    std::cout << "end test.\n";
}

// TEST_P(VencTest, NvstreamStreaming) {
//     EXPECT_TRUE(handle_ != nullptr);
//     EXPECT_TRUE(LoadCfg());
//     std::string cmd("/app/runtime_service/nvs_producer/bin/nvs_producer");
//     std::future<void> res = std::async(std::launch::async, StartProcess, cmd);
//     EXPECT_TRUE(Init());

//     SetCheckImgCb([](uint32_t size) { EXPECT_GT(size, 0); });

//     while (check_count_ < 100) {
//         std::cout << "check processing (" << check_count_ << "/100)\n";
//     }

//     if (TerminateProcessByName("nvs_producer")) {
//         std::cout << "exit nvs success!\n";
//     }
//     res.get();
//     EXPECT_EQ(0, 0);
// }

INSTANTIATE_TEST_CASE_P(MainInst, VencTest, ::testing::Values(&g_yuv_receiver));

// int main(int argc, char* argv[]) {
//     int aaa = 10;
//     std::cout << aaa << std::endl;
//     testing::InitGoogleTest(&argc, argv);
//     int res = RUN_ALL_TESTS();
//     return res;
// }
