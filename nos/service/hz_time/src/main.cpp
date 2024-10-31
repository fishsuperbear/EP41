#include <signal.h>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include "em/include/exec_client.h"
#include "hz_time.h"

std::mutex g_mutex;
std::condition_variable g_cv;
bool g_need_stop = false;

#define CONFIG_FILE_PATH "conf/hz_time.yaml"

using namespace hozon::netaos::hz_time;

void SigHandler(int signum) {
    TIME_EARLY_LOG << "hz_time signal enter, signum [" << signum << "]";
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
    TIME_EARLY_LOG << "config file path: " << cm_conf_path;

    YAML::Node config = YAML::LoadFile(cm_conf_path);
    auto server_ip = config["server_ip"].as<std::string>();
    auto log_level = config["logLevel"].as<uint8_t>();
    auto log_mode = config["logMode"].as<uint8_t>();
    auto file = config["file"].as<std::string>();

    hozon::netaos::hz_time::Time::GetInstance().SetParameter(server_ip);
    hozon::netaos::hz_time::TimeLogger::GetInstance().InitLogger(static_cast<hozon::netaos::log::LogLevel>(log_level), log_mode, file);
    hozon::netaos::hz_time::TimeLogger::GetInstance().CreateLogger(static_cast<hozon::netaos::log::LogLevel>(log_level));

    std::unique_ptr<hozon::netaos::em::ExecClient> exe_client(new hozon::netaos::em::ExecClient());
    TIME_LOG_INFO << "Em report running state:" << exe_client->ReportState(hozon::netaos::em::ExecutionState::kRunning);

    TIME_LOG_INFO << "Before Init";
    TIME_LOG_INFO << "Init status:" << hozon::netaos::hz_time::Time::GetInstance().Init();
    TIME_LOG_INFO << "After Init";

    while (!g_need_stop) {
        std::unique_lock<std::mutex> lock(g_mutex);
        g_cv.wait(lock);
    }

    TIME_LOG_INFO << "Before Deinit";
    TIME_LOG_INFO << "Deinit status:" << hozon::netaos::hz_time::Time::GetInstance().Deinit();
    TIME_LOG_INFO << "After Deinit";

    TIME_LOG_INFO << "Em report terminating state:" << exe_client->ReportState(hozon::netaos::em::ExecutionState::kTerminating);

    return 0;
}