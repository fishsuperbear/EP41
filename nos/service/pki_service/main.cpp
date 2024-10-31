#include <csignal>
#include <thread>
#include <iostream>
#include <chrono>
// #include "hz_fm_agent.h"
#include "sys/stat.h"
#include "unistd.h"
#include "yaml-cpp/yaml.h"
#include "tsp_pki_log.h"
#include "tsp_pki_cert_manage.h"
#include "tsp_pki_config.h"
#include "tsp_pki_utils.h"
#include "https.h"
#include "em/include/exec_client.h"
#include "em/include/proctypes.h"
#include "phm/include/phm_client.h"

sig_atomic_t g_stopFlag = 0;
using namespace hozon::netaos::em;
using namespace hozon::netaos::phm;
using namespace  hozon::netaos::tsp_pki;

void SigHandler(int signum)
{

    PKI_INFO << "Received signal: " << signum  << ". Quitting.";
    signal(SIGINT, SIG_IGN);
    signal(SIGTERM, SIG_IGN);
    signal(SIGQUIT, SIG_IGN);
    signal(SIGHUP, SIG_IGN);

    g_stopFlag = true;

    signal(SIGINT, SIG_DFL);
    signal(SIGTERM, SIG_DFL);
    signal(SIGQUIT, SIG_DFL);
    signal(SIGHUP, SIG_DFL);
}

/** Sets up signal handler.*/
void SigSetup(void)
{
    struct sigaction action
    {
    };
    action.sa_handler = SigHandler;

    sigaction(SIGINT, &action, nullptr);
    sigaction(SIGTERM, &action, nullptr);
    sigaction(SIGQUIT, &action, nullptr);
    sigaction(SIGHUP, &action, nullptr);
}

int main(int argc, char* argv[]){
    // Register signal hanlder for SIGTERM(EM Termination) and SIGINT(ctrl+c)
    SigSetup();

    TspPKILog::GetInstance().setLogLevel(static_cast<int32_t>(TspPKILog::CryptoLogLevelType::CRYPTO_INFO));
    TspPKILog::GetInstance().InitLogging("pki_service", "pki service",
                                                TspPKILog::CryptoLogLevelType::CRYPTO_INFO,                  //the log level of application
                                                hozon::netaos::log::HZ_LOG2FILE,  //the output log mode
                                                "/opt/usr/log/soc_log/",                                                                  //the log file directory, active when output log to file
                                                10,                                                                    //the max number log file , active when output log to file
                                                20                                                                     //the max size of each  log file , active when output log to file
    );
    TspPKILog::GetInstance().CreateLogger("PKI");
    PKI_INFO<< "TspPKILog init finish.";

    std::shared_ptr<ExecClient> execlient = std::make_shared<ExecClient>();
    auto ret_em = execlient->ReportState(ExecutionState::kRunning);
    if (ret_em) {
        PKI_INFO << "pki_service report em status failed." ;
    }

    // std::shared_ptr<PHMClient> spPHMClient(new PHMClient());
    // spPHMClient->Init("", nullptr, nullptr);

    std::string config_path = hozon::netaos::tsp_pki::TspPkiUtils::GetAppDirectory();
    config_path += "/conf/pki_service.yaml";
    const std::string TSP_PKI_CONFIG_USER_DEBUG = "/cfg/pki/conf/pki_service.yaml";
    if (hozon::netaos::tsp_pki::TspPkiUtils::IsFileExist(TSP_PKI_CONFIG_USER_DEBUG)) {
        config_path = TSP_PKI_CONFIG_USER_DEBUG;
        PKI_WARN << "use TSP_PKI_CONFIG_USER_DEBUG :" <<TSP_PKI_CONFIG_USER_DEBUG;
    }
    PKI_INFO <<"config_path: "<< config_path;

    if (0 != access(config_path.c_str(), F_OK | R_OK)) {
        // SendFault_t cFault(FmFaultCode::FmCfgFileError, 1, 1);
        // spPHMClient->ReportFault(cFault);
        PKI_INFO<<config_path<<" do not exist.";
        return 0;
    }

    std::this_thread::sleep_for(std::chrono::seconds(2));
    hozon::netaos::tsp_pki::TspPkiConfig::Instance().Start(config_path);
    hozon::netaos::tsp_pki::TspPkiCertManage::Instance().Start();

    while (!g_stopFlag) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    PKI_INFO << "Stop tsp pki service.";
    // Stop all sub-module and release resources.

    // PKI_INFO << "Stop tsp pki cert manager.";
    hozon::netaos::tsp_pki::TspPkiCertManage::Instance().Stop();
    hozon::netaos::tsp_pki::TspPkiCertManage::Destroy();

    // PKI_INFO << "Stop tsp pki config.";
    hozon::netaos::tsp_pki::TspPkiConfig::Instance().Stop();
    hozon::netaos::tsp_pki::TspPkiConfig::Destroy();

    // spPHMClient->Deinit();

    PKI_INFO << "Exit main.";
    execlient->ReportState(ExecutionState::kTerminating);
    return 0;
}
