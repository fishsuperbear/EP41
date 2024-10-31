#include <iostream>
#include <memory>
#include <unistd.h>
#include <csignal>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "pub.h"
#include "sub.h"

using namespace hozon::netaos::cm;
bool g_stopFlag = false;
std::mutex mtx;
std::condition_variable cv;

void INTSigHandler(int32_t num)
{
    (void)num;

    g_stopFlag = true;
    std::unique_lock<std::mutex> lck(mtx);
    cv.notify_all();
}


int main(
        int argc,
        char** argv)
{
    /*Need add SIGTERM from EM*/
    signal(SIGTERM, INTSigHandler);
    signal(SIGINT, INTSigHandler);
    signal(SIGPIPE, SIG_IGN);

    int type = 0;

    if (argc == 2)
    {
        if (strcmp(argv[1], "publisher") == 0)
        {
            type = 1;
        }
        else if (strcmp(argv[1], "subscriber") == 0)
        {
            type = 2;
        }
    }

    if (type == 0)
    {
        std::cout << "Error: Incorrect arguments." << std::endl;
        std::cout << "Usage: " << std::endl << std::endl;
        std::cout << argv[0] << " publisher|subscriber" << std::endl << std::endl;
        return 0;
    }

    DefaultLogger::GetInstance().InitLogger();
    DF_LOG_INFO << "Start..";

    pub pub_;
    HelloWorldSubscriber sub_;
    std::thread th_sub_;
    std::thread th_pub_;
    // Register the type being used
    switch (type)
    {
        case 1:
        {
            pub_.init("avmTopic111");
            th_pub_ = std::thread([&]() { pub_.run(); });
            break;
        }
        case 2:
        {
            sub_.init();
            th_sub_ = std::thread([&]() { sub_.run(); });
            break;
        }
    }

    while (!g_stopFlag) {
        std::unique_lock<std::mutex> lck(mtx);
        cv.wait(lck);

        switch (type)
        {
            case 1:
            {
                pub_.deinit();
                th_pub_.join();
                break;
            }
            case 2:
            {
                sub_.deinit();
                th_sub_.join();
                break;
            }
        }
    }

    return 0;
}
