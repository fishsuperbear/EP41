#include <time.h>
#include <signal.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
#include "argvparser.h"
#include "cfg/include/config_param.h"

using namespace argvparser;

bool stopped_ = false;

struct TsyncOptions {
    std::string time_type = "";  //DP\DP_AND_MP\MP
    bool set_option = false;
    bool continuous = false;
    bool set_mp_offset = 0;
    std::vector<std::string> set_values;
};

void SignalHandler(int signum) {
    stopped_ = true;
}

void PrintData(timespec& ts) {
    std::chrono::seconds sec(ts.tv_sec);        // 秒
    std::chrono::nanoseconds nsec(ts.tv_nsec);  // 纳秒
    // 将时间段相加
    std::chrono::steady_clock::time_point time_point = std::chrono::steady_clock::time_point(sec) + nsec;
    // 将time_point转换为std::time_t
    std::time_t time_t_value = std::chrono::system_clock::to_time_t(std::chrono::system_clock::time_point() + std::chrono::duration_cast<std::chrono::seconds>(time_point.time_since_epoch()));
    std::tm tmTime = *std::localtime(&time_t_value);
    // 格式化为字符串
    std::ostringstream oss;
    oss << std::put_time(&tmTime, "%a %b %d %H:%M:%S %Y");
    std::cout << oss.str() << std::endl;
}

void GetTime(std::string time_type) {
    struct timespec ts;
    if ("DP" == time_type || "DP_AND_MP" == time_type) {
        if (clock_gettime(CLOCK_REALTIME, &ts)) {
            std::cout << "get dp time wrong" << std::endl;
            return;
        }
        std::cout << "DP:" << ts.tv_sec << " sec, " << ts.tv_nsec << " nsec." << std::endl;
        PrintData(ts);
    }
    if ("MP" == time_type || "DP_AND_MP" == time_type) {
        if (clock_gettime(12, &ts)) {
            std::cout << "";
            return;
        }

        std::cout << "MP:" << ts.tv_sec << " sec, " << ts.tv_nsec << " nsec." << std::endl;
        PrintData(ts);
    }
}

void SetTime(TsyncOptions tsync_options) {
    if (tsync_options.set_values.size() != 2) {
        std::cout << "Parameter error, use - h to print the help infomation" << std::endl;
        return;
    }
    //设置时间
    try {
        struct timespec time = {std::stoll(tsync_options.set_values[0]), std::stoll(tsync_options.set_values[1])};
        if ("DP" == tsync_options.time_type) {
            int ret_set_time = clock_settime(CLOCK_REALTIME, &time);
            if (ret_set_time != 0) {
                std::cout << "set dp time wrong";
                GetTime("DP");
            }
        } else if ("MP" == tsync_options.time_type) {

            while (!stopped_) {
                int ret_set_time = clock_settime(12, &time);
                if (ret_set_time != 0) {
                    std::cout << "set mp time wrong";
                    GetTime("MP");
                }

                if (tsync_options.continuous) {
                    std::cout << "Set management time to " << time.tv_sec << "." << time.tv_nsec << std::endl;
                }
                else {
                    break;
                }

                std::this_thread::sleep_for(std::chrono::microseconds(999800));
                time.tv_sec += 1;
            }
        } else {
            std::cout << "Parameter error, use - h to print the help infomation" << std::endl;
        }
    } catch (const std::invalid_argument& e) {
        std::cerr << "Invalid argument: " << e.what() << std::endl;
        return;
    } catch (const std::out_of_range& e) {
        std::cerr << "Out of range: " << e.what() << std::endl;
        return;
    }
}

void HandleTime(TsyncOptions tsync_options) {

#ifdef TARGET_PLATFORM
    std::string platform = TARGET_PLATFORM;
    if ("orin" == platform) {
        if (tsync_options.set_option) {

            if (tsync_options.time_type.find("MP") != std::string::npos) {
                struct timespec ts {0};
                if (clock_gettime(CLOCK_REALTIME, &ts)) {
                    std::cout << "get dp time wrong" << std::endl;
                    return;
                }
                int64_t mp_offset = std::stoll(tsync_options.set_values[0]) - ts.tv_sec;

                if (hozon::netaos::cfg::CfgResultCode::CONFIG_OK == hozon::netaos::cfg::ConfigParam::Instance()->Init(2000)) {
                    uint8_t manual = 1;
                    if (hozon::netaos::cfg::CfgResultCode::CONFIG_OK != hozon::netaos::cfg::ConfigParam::Instance()->SetParam<uint8_t>("time/manual_sync", manual)) {
                        std::cout << "Warn: cannot set config: time/manual_sync. Manangement time may be over written by hz_time process\n";
                    }
                    if (tsync_options.set_mp_offset) {

                        if (hozon::netaos::cfg::CfgResultCode::CONFIG_OK != hozon::netaos::cfg::ConfigParam::Instance()->SetParam<int64_t>("time/mp_offset", mp_offset)) {
                            std::cout << "Warn: cannot set time/mp_offset.\n";
                        }
                    }
                    
                    // Sleep to make sure that time_sync process has enough time to be notified the state of time/manual_sync.
                    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
                }
            }

            SetTime(tsync_options);

            if (tsync_options.time_type.find("MP") != std::string::npos) {
                hozon::netaos::cfg::ConfigParam::Instance()->DeInit();
            }
        } else {
            GetTime(tsync_options.time_type);
        }
    } else {
        std::cout << "tsync commend only support in orin platform" << std::endl;
    }
#else
    std::cout << "tsync commend only support in orin platform" << std::endl;
#endif
}

int main(int argc, char** argv) {

    signal(SIGINT, SignalHandler);

    struct TsyncOptions tsync_options;
    bool dp_show_help_info = false;
    bool mp_show_help_info = false;
    bool dp_mp_show_help_info = false;

    auto dp_argv_parser =   (
                                command("DP").set(tsync_options.time_type, std::string("DP")),
                                (required("-h", "--help").set(dp_show_help_info, true) % "print the help infomation") |
                                (required("-g").set(tsync_options.set_option, false) % "get data plane time.") |
                                (required("-s").set(tsync_options.set_option, true) % "set data plane time. eg: tsync DP -s 1639120283 0" & values("time: sec nsec", tsync_options.set_values))
                            );

    auto mp_argv_parser =   (
                                command("MP").set(tsync_options.time_type, std::string("MP")),
                                (
                                    required("-h", "--help").set(mp_show_help_info, true) % "print the help infomation"
                                ) |
                                (
                                    required("-g").set(tsync_options.set_option, false) % "get management plane time."
                                ) |
                                (
                                    (
                                        required("-s").set(tsync_options.set_option, true) % "set manage plane time. eg: tsync MP -s 1639120283 0" & values("time: sec nsec", tsync_options.set_values)
                                    ),
                                    (
                                        option("-c").set(tsync_options.continuous, true) % "continuouly set time (only for management plane)"
                                    ),
                                    (
                                        option("-d").set(tsync_options.set_mp_offset, true) % "for data_collection: calculate mp time by using offset from data time."
                                    )
                                )
                            );

    auto dp_mp_argv_parser =    (
                                    command("DP_AND_MP").set(tsync_options.time_type, std::string("DP_AND_MP")),
                                    (required("-h", "--help").set(dp_mp_show_help_info, true) % "print the help infomation") |
                                    (required("-g").set(tsync_options.set_option, false) % "get manage plane time.") |
                                    (required("-s").set(tsync_options.set_option, true)  % "set data / manage plane time. eg: tsync DP_AND_MP -s 1639120283 0" & values("time: sec nsec", tsync_options.set_values))
                                );

    auto tsync_argv_parser = (dp_argv_parser | mp_argv_parser | dp_mp_argv_parser);

    if (parse(argc, argv, tsync_argv_parser)) {
        if (dp_show_help_info) {
            std::cout << make_man_page(dp_argv_parser, argv[0]) << '\n';
        } else if (mp_show_help_info) {
            std::cout << make_man_page(mp_argv_parser, argv[0]) << '\n';
        } else if (dp_mp_show_help_info) {
            std::cout << make_man_page(dp_mp_argv_parser, argv[0]) << '\n';
        } else if ("DP" == tsync_options.time_type || "MP" == tsync_options.time_type || "DP_AND_MP" == tsync_options.time_type) {
            HandleTime(tsync_options);
        } else {
            std::cout << "Parameter error, use - h to print the help infomation" << '\n';
        }
    } else {
        // std::cout << "Supported sub-command: DP MP DP_AND_MP. \nUse -h on sub-command to get help information.\n";

        man_page man;
        man.append_section("SYNOPSIS", usage_lines(tsync_argv_parser,  argv[0], doc_formatting{}).str());
        man.append_section("OPTIONS (DP)", documentation(dp_argv_parser, doc_formatting{}).str());
        man.append_section("OPTIONS (MP)", documentation(mp_argv_parser, doc_formatting{}).str());
        man.append_section("OPTIONS (DP_AND_MP)", documentation(dp_mp_argv_parser, doc_formatting{}).str());
        std::cout << man << '\n';
    }
    return 0;
}