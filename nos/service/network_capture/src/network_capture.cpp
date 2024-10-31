#include "json/json.h"
#include <netinet/ip.h>
#include "network_capture/include/network_capture.h"
#include "network_capture/include/network_logger.h"

/*debug*/#include "network_capture/include/function_statistics.h"
namespace hozon {
namespace netaos {
namespace network_capture {

std::int32_t NetworkCapture::Init() {
    std::string configFile = std::string(CONFIG_PATH) + "/common_config.json";
    if (0 == access(configFile.c_str(), F_OK)) {
        Json::Value rootReder;
        Json::CharReaderBuilder readBuilder;
        std::ifstream ifs(configFile);
        std::unique_ptr<Json::CharReader> reader(readBuilder.newCharReader());
        JSONCPP_STRING errs;
        if (Json::parseFromStream(readBuilder, ifs, &rootReder, &errs)) {
            someip_flag_ = (0 != rootReder["SomeipFlag"].asBool()) ? rootReder["SomeipFlag"].asBool() : someip_flag_;
            lidar_cloudpoint_flag_ = (0 != rootReder["CloudpointFlag"].asBool()) ? rootReder["CloudpointFlag"].asBool() : lidar_cloudpoint_flag_;
        }
    } else {
        NETWORK_LOG_ERROR << "load common_config.json error";
    }

    lidar_mtx_ = std::make_shared<std::mutex>();
    someip_mtx_ = std::make_shared<std::mutex>();

    if (lidar_cloudpoint_flag_) {
        lidar_pub_list_ = std::make_shared<std::queue<std::unique_ptr<hozon::soc::RawPointCloud>>>();
        lidar_capture_config_ = LidarFilterInfo::LoadConfig();
        lidar_capture_ = std::make_unique<LidarCapture>(*lidar_capture_config_, lidar_pub_list_, lidar_mtx_);
        lidar_capture_->Init();
    }

    std::map<std::uint32_t, std::string> topic_map;
    if (someip_flag_) {
        someip_pub_list_ = std::make_shared<std::queue<std::unique_ptr<raw_someip_message>>>();
        someip_capture_config_list_ = SomeipFilterInfo::LoadConfig(CONFIG_PATH);

        for (const auto &config : *someip_capture_config_list_) {
            for (const auto &topic : config->topic_map) {
                topic_map[topic.first] = topic.second;
            }
            auto someip_capture_ptr = std::make_unique<SomeipCapture>(*config, someip_pub_list_, someip_mtx_);
            someip_capture_list_.emplace_back(std::move(someip_capture_ptr));
        }

        for (const auto &someip_capture_ : someip_capture_list_) {
            someip_capture_->Init();
        }
    }

    network_pub_ = std::make_unique<NetworkPub>(lidar_pub_list_, lidar_mtx_, someip_pub_list_, someip_mtx_, topic_map);
    network_pub_->Init();

    NETWORK_LOG_INFO << "SomeipFlag : " << someip_flag_;
    NETWORK_LOG_INFO << "CloudpointFlag : " << lidar_cloudpoint_flag_;
    return true;
}

std::int32_t NetworkCapture::Run() {
    NETWORK_LOG_DEBUG << "NetworkCapture::Run";
    if (someip_flag_ || lidar_cloudpoint_flag_)
        network_pub_->Run();

    if (lidar_cloudpoint_flag_) {
        lidar_run_thread_ = std::make_unique<std::thread>([&]() -> void {
            bool lidar_state = false;
            while (!stop_flag_) {
                if (network_pub_->Lidar_IsMatched() != lidar_state) {
                    if (false == lidar_state) {
                        lidar_capture_->Run();
                    } else {         
                        lidar_capture_->Stop();
                    }
                    lidar_state = !lidar_state;
                    network_pub_->Set_Lidar_State(lidar_state);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                NETWORK_LOG_INFO << "lidar_topic_match_state: " << lidar_state;
            }

            if (lidar_state) 
                lidar_capture_->Stop();
        });
    }

    if (someip_flag_) {
        someip_run_thread_ = std::make_unique<std::thread>([&]() -> void {
            bool someip_state = false;
            while (!stop_flag_) {
                if (network_pub_->someip_IsMatched() != someip_state) {
                    if (false == someip_state) {
                        for (const auto &someip_capture_ : someip_capture_list_) {
                            someip_capture_->Run();
                        }
                    } else {         
                        for (const auto &someip_capture_ : someip_capture_list_) {
                            someip_capture_->Stop();
                        }
                    }
                    someip_state = !someip_state;
                    network_pub_->Set_Someip_State(someip_state);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                NETWORK_LOG_INFO << "someip_topic_match_state : " << someip_state;
            }

            if (someip_state) {
                for (const auto &someip_capture_ : someip_capture_list_) {
                    someip_capture_->Stop();
                }
            }
        });
    }

    return true;
}

std::int32_t NetworkCapture::Stop() {
    FunctionStatistics("NetworkCapture::Stop");
    stop_flag_ = true;
    
    if (lidar_cloudpoint_flag_ && lidar_run_thread_->joinable())
        lidar_run_thread_->join();

    if (someip_flag_ && someip_run_thread_->joinable())
        someip_run_thread_->join();

    if (someip_flag_ || lidar_cloudpoint_flag_)
        network_pub_->Stop();
    return true;
}

std::int32_t NetworkCapture::DeInit() {
    if (someip_flag_ || lidar_cloudpoint_flag_)
        network_pub_->Deinit();
    return true;
}


}  // namespace network_capture
}  // namespace netaos
}  // namespace hozon
