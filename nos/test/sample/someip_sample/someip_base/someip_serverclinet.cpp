
#include "someip_impl.h"
#include <signal.h>
#include <stdio.h>
#include <thread>

bool stop_flag = false;
void SigHandler(int signum)
{
    printf("SigHandler signum: %d\n", signum);
    stop_flag = true;
}

int32_t main(int32_t argc, char** argv)
{
    signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);

    SomeipImpl *process = nullptr;
    std::string config = "conf/someip_config.json";
    if (argc >= 2) {
        config = argv[1];
        printf("config file: %s\n", config.c_str());
    }

    process = new SomeipImpl(config);
    process->Init();
    process->Start();

    while (!stop_flag) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    process->Stop();
    process->Deinit();

    return 0;
}