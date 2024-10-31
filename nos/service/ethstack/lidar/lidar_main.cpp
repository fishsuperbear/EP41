#include <csignal>
#include <memory>
#include <vector>
#include <string>
// #include <io.h>
#include <thread>
#include <unistd.h>

#include "yaml-cpp/yaml.h"
#include "common/logger.h"
#include "management/process_manager.h"
#include "faultmessage/lidar_fault_report.h"
#include "em/include/exec_client.h"
#include "cfg/include/config_param.h"

using namespace hozon::netaos::em;

#define CONFIG_FILE_PATH "/app/runtime_service/neta_lidar/conf/neta_lidar.yaml"

sig_atomic_t g_stopFlag = 0;
std::mutex mtx;
std::condition_variable cv;


void INTSigHandler(int32_t num)
{
    (void)num;
    g_stopFlag = 1;
    std::unique_lock<std::mutex> lck(mtx);
    cv.notify_all();
}

// int32_t main(int32_t argc, char* argv[])
int32_t main(int32_t argc, char* argv[])
{
    /*Need add SIGTERM from EM*/
    signal(SIGTERM, INTSigHandler);
    signal(SIGINT, INTSigHandler);
    signal(SIGPIPE, SIG_IGN);

    // if (argc != 3) {
    //     return  -1;
    // }

    std::string lidarpos,ifname;
    uint8_t logLevel = 2 ;
    uint8_t logMode = 2 ;  

    YAML::Node config;
    if (0 == access(CONFIG_FILE_PATH, F_OK)) {
        config = YAML::LoadFile(CONFIG_FILE_PATH);
        ifname = config["ifname"].as<std::string>();
        lidarpos = config["lidarpos"].as<std::string>();
        logLevel = config["logLevel"].as<uint32_t>();
        logMode = config["logMode"].as<uint32_t>();
    }

    LidarLogger::GetInstance().InitLogger(logLevel,logMode);
    LIDAR_LOG_INFO << "Lidar pos is " << lidarpos;
    LIDAR_LOG_INFO << "IP address is " << ifname;
    LIDAR_LOG_INFO << "Hozon lidar process start...~~~.";

    std::shared_ptr<ExecClient> execli(new ExecClient());
    int32_t ret = execli->ReportState(ExecutionState::kRunning);
    if(ret){ LIDAR_LOG_INFO << "EM neta_lidar report fail....~~~.";}

    hozon::ethstack::lidar::LidarFaultReport::Instance().Init();
    hozon::netaos::cfg::ConfigParam::Instance()->Init(3000);
    hozon::ethstack::lidar::ProcessManager::Instance().SetIfName(ifname);
    hozon::ethstack::lidar::ProcessManager::Instance().SetLidarFrame(lidarpos);
    hozon::ethstack::lidar::PointCloudParser::Instance().Init();
    
    hozon::ethstack::lidar::ProcessManager::Instance().Init();
    hozon::ethstack::lidar::ProcessManager::Instance().Start();

    while (!g_stopFlag) {
        std::unique_lock<std::mutex> lck(mtx);
        cv.wait(lck);
    }

    hozon::ethstack::lidar::ProcessManager::Instance().Stop();
    hozon::ethstack::lidar::LidarFaultReport::Instance().DeInit();
    hozon::netaos::cfg::ConfigParam::Instance()->SetParam("system/lidar_status", (uint8_t)2);
    hozon::netaos::cfg::ConfigParam::Instance()->DeInit();

    LIDAR_LOG_INFO << "Hozon lidar process stopped!~~~.";
    return 0;
}
