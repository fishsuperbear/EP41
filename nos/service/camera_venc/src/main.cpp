#include <getopt.h>
#include <signal.h>
#include <string>
#include <thread>
#include "camera_venc.h"
#include "em/include/exec_client.h"
#include "em/include/proctypes.h"

using namespace hozon::netaos::em;

static hozon::netaos::cameravenc::CameraVenc app_;
static bool stopped_ = false;

bool g_write_to_file = false;
int8_t g_consumer_index = 0;
std::string g_conf_path = "/app/runtime_service/camera_venc/conf/camera_venc_conf.yaml";

/** Signal handler.*/
static void SigHandler(int signum) {

    std::cout << "Received signal: " << signum << ". Quitting\n";
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
static void SigSetup(void) {
    struct sigaction action {};

    action.sa_handler = SigHandler;

    sigaction(SIGINT, &action, nullptr);
    sigaction(SIGTERM, &action, nullptr);
    sigaction(SIGQUIT, &action, nullptr);
    sigaction(SIGHUP, &action, nullptr);
}

int main(int argc, char* argv[]) {
    SigSetup();

    const struct option long_options[] = {{"consumer-index", required_argument, nullptr, 'c'},  //
                                          {"write", no_argument, nullptr, 'w'},                 //
                                          {nullptr, 0, nullptr, 0}};

    while (1) {
        int option_index = 0;
        int c = getopt_long(argc, argv, "c:wp:", long_options, &option_index);

        bool parse_end = false;
        switch (c) {
            case 'c':
                g_consumer_index = std::stoul(optarg);
                break;
            case 'w':
                g_write_to_file = true;
                break;
            case 'p':
                g_conf_path = "";
                g_conf_path.assign(optarg);
                std::cout << "conf path = " << g_conf_path << std::endl;
                break;
            case -1:
                parse_end = true;
                break;
            default:
                std::cout << argv[1] << " [-c <consumer_index 0|1|2>] [-w]" << std::endl;
                break;
        }

        if (parse_end) {
            break;
        }
    }

    std::shared_ptr<ExecClient> execli = std::make_shared<ExecClient>();
    execli->ReportState(ExecutionState::kRunning);

    app_.Init();
    app_.Start();

    do {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (!stopped_);

    app_.Stop();
    app_.Deinit();

    execli->ReportState(ExecutionState::kTerminating);
    return 0;
}