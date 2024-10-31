#include <signal.h>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include "em/include/exec_client.h"
#include "hz_dvr.h"
#include "logger.h"

std::mutex g_mutex;
std::condition_variable g_cv;
bool g_need_stop = false;

#define CONFIG_FILE_PATH "conf/hz_dvr.yaml"

using namespace hozon::netaos::hz_dvr;

void SigHandler(int signum) {
    hozon::netaos::hz_dvr::DvrEarlyLogger() << "hz_dvr signal enter, signum [" << signum << "]";
    g_need_stop = true;
    std::unique_lock<std::mutex> lock(g_mutex);
    g_cv.notify_all();
}

int main(int argc, char* argv[]) {
    signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);

    std::string binary_path = std::string(argv[0]);
    size_t pos = 0;
    for (size_t i = 0; i < binary_path.size(); i++) {
        if (binary_path[i] == '/') {
            pos = i;
        }
    }
    std::string folder_path = binary_path.substr(0, pos);
    std::string cm_conf_path = folder_path + "/../" + std::string(CONFIG_FILE_PATH);
    hozon::netaos::hz_dvr::DvrEarlyLogger() << "config file path: " << cm_conf_path;

    YAML::Node config = YAML::LoadFile(cm_conf_path);
    auto width = config["width"].as<uint32_t>();
    auto height = config["height"].as<uint32_t>();
    auto sensor_id = config["sensor_id"].as<uint32_t>();
    auto channel_id = config["channel_id"].as<uint32_t>();
    auto log_level = config["logLevel"].as<uint8_t>();
    auto log_mode = config["logMode"].as<uint8_t>();
    auto file = config["file"].as<std::string>();

    hozon::netaos::hz_dvr::DvrEarlyLogger() << "width:" << width;
    hozon::netaos::hz_dvr::DvrEarlyLogger() << "height:" << height;
    hozon::netaos::hz_dvr::DvrEarlyLogger() << "sensor_id:" << sensor_id;
    hozon::netaos::hz_dvr::DvrEarlyLogger() << "channel_id:" << channel_id;

    hozon::netaos::hz_dvr::Dvr::GetInstance().SetParameter(width, height, sensor_id, channel_id);
    hozon::netaos::hz_dvr::DvrLogger::GetInstance().InitLogger(static_cast<hozon::netaos::log::LogLevel>(log_level),
                                                               log_mode, file);
    hozon::netaos::hz_dvr::DvrLogger::GetInstance().CreateLogger(static_cast<hozon::netaos::log::LogLevel>(log_level));

    std::unique_ptr<hozon::netaos::em::ExecClient> exe_client(new hozon::netaos::em::ExecClient());
    hozon::netaos::hz_dvr::DvrEarlyLogger()
        << "Em report running state:" << exe_client->ReportState(hozon::netaos::em::ExecutionState::kRunning);

    DVR_LOG_INFO << "Before Init";
    hozon::netaos::hz_dvr::Dvr::GetInstance().Init();
    DVR_LOG_INFO << "After Init";

    DVR_LOG_INFO << "Before Run";
    hozon::netaos::hz_dvr::Dvr::GetInstance().Run();
    DVR_LOG_INFO << "After Run";

    while (!g_need_stop) {
        std::unique_lock<std::mutex> lock(g_mutex);
        g_cv.wait(lock);
    }

    DVR_LOG_INFO << "Before Deinit";
    hozon::netaos::hz_dvr::Dvr::GetInstance().Deinit();
    DVR_LOG_INFO << "After Deinit";

    hozon::netaos::hz_dvr::DvrEarlyLogger()
        << "Em report terminating state:" << exe_client->ReportState(hozon::netaos::em::ExecutionState::kTerminating);

    return 0;
}