#include <iostream>
#include <memory>
#include <unistd.h>
#include <csignal>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "app_pub.h"
#include "app_sub.h"

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
    int control_interval = 100;
    bool control_flag = false;

    if (argc >= 2)
    {
        if (strcmp(argv[1], "pub") == 0)
        {
            type = 1;
            if (argc == 3) {
                std::string interval(argv[2]);
                control_interval = std::stoi(interval);
                control_flag = true;
            }
        }
        else if (strcmp(argv[1], "sub") == 0)
        {
            type = 2;
        }
    }
    
    

    if (type == 0)
    {
        std::cout << "Error: Incorrect arguments." << std::endl;
        std::cout << "Usage: " << std::endl << std::endl;
        std::cout << argv[0] << " pub | sub" << std::endl << std::endl;
        return 0;
    }


    app_pub pub_;
    app_sub sub_;
    std::thread th_sub_;
    std::thread th_pub_;
    // Register the type being used
    switch (type)
    {
        case 1:
        {
            pub_.init("avmTopic");
            th_pub_ = std::thread([&]() { pub_.run(control_interval); });

            if (control_flag) {
WAIT:
                std::string str_argv_input;
                do {
                    getline(cin, str_argv_input);
                } while (str_argv_input.empty());

                int control_interval_tmp = stoi(str_argv_input);
                if (0 == control_interval_tmp) {
                    goto QUIT;
                }

                pub_.set_interval(control_interval_tmp);
                std::cout << "change send period: " << control_interval_tmp << " ms. is ok!" << std::endl;
                goto WAIT;
            }
            break;
        }
        case 2:
        {
            hozon::netaos::log::InitLogging(
                "PHM_APP",
                "PHM_APP",
                hozon::netaos::log::LogLevel::kInfo,
                hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE,
                "/log/",
                10,
                100
            );

            sub_.init("avmTopic");
            th_sub_ = std::thread([&]() { sub_.run(std::bind(&app_sub::sub_callback, &sub_)); });
            break;
        }
    }

QUIT:

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
