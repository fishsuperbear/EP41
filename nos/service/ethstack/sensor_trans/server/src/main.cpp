

#include <condition_variable>
#include <csignal>
#include <iostream>
#include <memory>
#include <mutex>
#include <unistd.h>
#include "sensor_manager.h"
#include "logger.h"
#include "config_loader.h"

#define CONFIG_FILE_PATH "conf/sensor_trans.yaml"

#define SENSOR_TRANS_CONFIG_FILE_DEFAULT ("/app/runtime_service/sensor_trans/conf/sensor_trans.yaml")

using namespace hozon::netaos::sensor;


// int g_stopFlag = 0;
// int g_signum = 0;
// void SigHandler(int signum) {
//     g_signum = signum;
//     execClient->ReportState(ExecutionState::kTerminating);
//     std::cout << "sensor trans main receive terminating signum." << std::endl;
//     g_stopFlag = 1;
// }


int main(int argc, char* argv[]) {
    // signal(SIGINT, SigHandler);
    // signal(SIGTERM, SigHandler);

    // std::string binary_path = std::string(argv[0]);
    // size_t pos = 0;
    // for(size_t i = 0; i < binary_path.size(); i++) {
    //     if(binary_path[i] == '/') {
    //         pos = i;
    //     }
    // }
    // std::string folder_path = binary_path.substr(0, pos);
    // std::string conf_path = folder_path + "/../" + std::string(CONFIG_FILE_PATH);

    std::string conf_path = SENSOR_TRANS_CONFIG_FILE_DEFAULT;
    hozon::netaos::sensor::ConfigLoader::LoadConfig(conf_path);

    hozon::netaos::sensor::Sensorlogger::GetInstance().InitLogger(
            static_cast<hozon::netaos::log::LogLevel>(ConfigLoader::log_level_), ConfigLoader::log_mode_,  ConfigLoader::log_file_);

    hozon::netaos::sensor::Sensorlogger::GetInstance().CreateLogger(static_cast<hozon::netaos::log::LogLevel>(ConfigLoader::log_level_));

    SENSOR_LOG_INFO << "sensor trans main begin...";
    std::shared_ptr<hozon::netaos::sensor::SensorManager> sensor_manager = std::make_shared<hozon::netaos::sensor::SensorManager>();
    sensor_manager->Init(conf_path, ConfigLoader::nnp_);

    // while (!g_stopFlag) {
    //     std::this_thread::sleep_for(std::chrono::milliseconds(500));
    // }

    SENSOR_LOG_INFO << "sensor_manager->WaitStop() ...";
    sensor_manager->WaitStop();
    SENSOR_LOG_INFO << "sensor_manager->Stop() ...";
    if (sensor_manager != nullptr) {
        sensor_manager->Stop();
    }
    SENSOR_LOG_INFO << "sensor trans main stop completed.";
    return 0;

}
