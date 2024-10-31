#include <signal.h>
#include <getopt.h>
#include <thread>
#include <string>
#include "crypto_server.h"
#include "idl/generated/crypto.h"
#include "idl/generated/cryptoPubSubTypes.h"
#include "common/crypto_logger.hpp"
#include "em/include/exec_client.h"
#include "em/include/proctypes.h"
#include "phm/include/phm_client.h"
#include "crypto_server_config.h"

#include "import_key.h"

static bool stopped_ = false;

/** Signal handler.*/
static void SigHandler(int signum)
{

    std::cout << "Crypto server received signal: " << signum  << ". Quitting\n";
    signal(SIGINT, SIG_IGN);
    signal(SIGTERM, SIG_IGN);
    signal(SIGQUIT, SIG_IGN);
    signal(SIGHUP, SIG_IGN);

    stopped_ = true;

    signal(SIGINT, SIG_DFL);
    signal(SIGTERM, SIG_DFL);
    signal(SIGQUIT, SIG_DFL);
    signal(SIGHUP, SIG_DFL);
}

/** Sets up signal handler.*/
static void SigSetup(void)
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
using namespace hozon::netaos::em;
using namespace hozon::netaos::phm;
using namespace hozon::netaos::crypto;

int main(int argc, char* argv[])
{
	SigSetup();

    CryptoLogger::GetInstance().setLogLevel(static_cast<int32_t>(CryptoLogger::CryptoLogLevelType::CRYPTO_INFO));
    CryptoLogger::GetInstance().InitLogging("crypto_server", "crypto_server",
                                                CryptoLogger::CryptoLogLevelType::CRYPTO_INFO,                  //the log level of application
                                                hozon::netaos::log::HZ_LOG2FILE,  //the output log mode
                                                "/opt/usr/log/soc_log/",                                                                  //the log file directory, active when output log to file
                                                10,                                                                    //the max number log file , active when output log to file
                                                20                                                                     //the max size of each  log file , active when output log to file
    );
    CryptoLogger::GetInstance().CreateLogger("CRYPS");

    CRYP_INFO<< "CryptoServerLog init finish.";

    std::shared_ptr<ExecClient> execlient = std::make_shared<ExecClient>();
    auto ret_em = execlient->ReportState(ExecutionState::kRunning);
    if (ret_em) {
        CRYP_INFO << "pki_service report em status failed.";
    }

    std::shared_ptr<PHMClient> spPHMClient(new PHMClient());
    spPHMClient->Init("", nullptr, nullptr);
    hozon::netaos::crypto::CryptoServer::Instance();
    hozon::netaos::crypto::CryptoServer::Instance().Init();
    hozon::netaos::crypto::CryptoServer::Instance().Start();

    ImportKey obj;
    std::string yamlfile = CryptoConfig::Instance().GetOemPresetKeyFile_En();
    obj.SaveKeyFromYaml(yamlfile);

    while (!stopped_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    hozon::netaos::crypto::CryptoServer::Instance().Stop();
    hozon::netaos::crypto::CryptoServer::Instance().Deinit();

    spPHMClient->Deinit();

    CRYP_INFO << "Exit main.";
    execlient->ReportState(ExecutionState::kTerminating);

    return 0;
}