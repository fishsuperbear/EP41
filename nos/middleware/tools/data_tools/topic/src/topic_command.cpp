#include <signal.h>
#include <string>
#include "argvparser.h"
#include "data_tools_logger.hpp"
#include "echo.h"
#include "hz.h"
#include "latency.h"
#include "list.h"
#include "monitor.h"

using namespace hozon::netaos::topic;
using namespace std;
using namespace argvparser;
#include <termio.h>

Echo* echo = nullptr;
Hz* hz = nullptr;
Monitor* monitor = nullptr;
Latency* latency = nullptr;
int type = 0;
bool stopped_ = false;

// 定义信号处理程序
void signalHandler(int signum) {

    stopped_ = true;
    // 终止程序
    if (type == 1) {
        if (nullptr != echo) {
            echo->Stop();
        }
    } else if (type == 2) {
        if (nullptr != hz) {
            hz->Stop();
        }
    } else if (type == 3) {
        if (nullptr != monitor) {
            monitor->Stop();
        }
    } else if (type == 4) {
        if (nullptr != latency) {
            latency->Stop();
        }
    }
}

// 函数用于设置终端的输入模式
void setTerminalInputMode(bool enableInput) {
    struct termios t;
    tcgetattr(0, &t);

    if (enableInput) {
        std::cout << "\033[?25h";  // 显示光标
        t.c_lflag |= ICANON | ECHO;
    } else {
        std::cout << "\033[?25l";  // 隐藏光标
        t.c_lflag &= ~(ICANON | ECHO);
    }
    tcsetattr(0, TCSANOW, &t);
}

void SigResizeHandle(int) {
    if (nullptr != monitor) {
        monitor->SigResizeHandle();
    }
};

int main(int argc, char** argv) {
    hozon::netaos::data_tool_common::BAGCONSOLELogger::GetInstance().setLogLevel(hozon::netaos::data_tool_common::LogLevel::kWarn);
    signal(SIGINT, signalHandler);
    signal(SIGHUP, signalHandler);
    signal(SIGTERM, signalHandler);
    signal(SIGWINCH, SigResizeHandle);

    signal(SIGUSR1, signalHandler);
    signal(SIGUSR2, signalHandler);
    signal(SIGPIPE, signalHandler);
    signal(SIGSTOP, signalHandler);
    signal(SIGTSTP, signalHandler);
    signal(SIGTTIN, signalHandler);
    signal(SIGTTOU, signalHandler);

    // signal(SIGCONT, signalHandler);
    // signal(SIGURG, signalHandler);
    // signal(SIGXCPU, signalHandler);
    // signal(SIGXFSZ, signalHandler);
    // signal(SIGVTALRM, signalHandler);
    // signal(SIGPROF, signalHandler);
    // signal(SIGSYS, signalHandler);
    // signal(SIGKILL, signalHandler);
    // signal(SIGABRT, signalHandler);

    // Log::SetVerbosity(Log::Info);
    std::string log_level = "";

    struct HzOptions hz_options;
    auto hz_argv_parser = (command("hz"), option("-h", "--help").set(hz_options.show_help_info, true) % "print the help infomation",
                           option("-a", "--all").set(hz_options.monitor_all, true) % " monitor all topics" | values("topic names", hz_options.events) % "specify the topic names",
                           option("-m", "--method").set(hz_options.method, true) % " show method topics",
                           (option("-s", "--skip_sample_num") & value("N", hz_options.skip_sample_num)) % "set the skip sample numbers when starting frequency statistics. Default 2.",
                           (option("-d", "--window_duration") & value("time", hz_options.window_duration)) % "Set the duration interval for frequency statistics. Unit: seconds. Default 0.",
                           (option("-e", "--exit_time") & value("time", hz_options.exit_time)) % "Set automatic exit time. Unit: seconds. Default 0",
                           (option("--log-level") & value("log-level", log_level)) % "set log level: [kDebug, kInfo, kWarn, kError]. Defaul 'kError'.");

    struct EchoOptions echo_options;
    auto echo_argv_parser =
        (command("echo"), option("-h", "--help").set(echo_options.show_help_info, true) % "print the help infomation" | values("topic name", echo_options.topics) % "specify the topic name",
         (option("--log-level") & value("log-level", log_level)) % "set log level: [kDebug, kInfo, kWarn, kError]. Defaul 'kError'.",
         (option("-c", "--someip_json_filepath") & value("./", echo_options.json_format_path)) % "Set someip json format file path. Default ./ ");

    struct MonitorOptions monitor_options;
    auto monitor_argv_parser = (command("monitor"), option("-h", "--help").set(monitor_options.show_help_info, true) % "print the help infomation",
                                option("-a", "--all").set(monitor_options.monitor_all, true) % " monitor all topics" | values("topic name", monitor_options.events) % "specify one topic name",
                                option("-m", "--method").set(monitor_options.method, true) % " show method topics",
                                (option("--log-level") & value("log-level", log_level)) % "set log level: [kDebug, kInfo, kWarn, kError]. Defaul 'kError'.");

    struct ListOptions list_options;
    bool list_show_help_info = false;
    bool list_execute = false;
    auto list_argv_parser = (command("list").set(list_execute, true), option("-h", "--help").set(list_show_help_info, true) % "print the help infomation",
                             option("-a", "--all").set(list_execute, true) % " show all topics", option("-m", "--method").set(list_options.method, true) % " show method topics");

    struct LatencyOptions latency_options;
    auto latency_argv_parser =
        (command("latency"), option("-h", "--help").set(latency_options.show_help_info, true) % "print the help information" | values("topic name", latency_options.topics) % "specify the topic name",
         (option("--log-level") & value("log-level", log_level)) % "set log level: [kDebug, kInfo, kWarn, kError]. Defaul 'kError'.");

    auto hztool_argv_parser = (echo_argv_parser | hz_argv_parser | monitor_argv_parser | latency_argv_parser | list_argv_parser);

    if (parse(argc, argv, hztool_argv_parser)) {
        if (hz_options.show_help_info) {
            cout << make_man_page(hz_argv_parser, argv[0]) << '\n';
            return 0;
        }
        if (echo_options.show_help_info) {
            cout << make_man_page(echo_argv_parser, argv[0]) << '\n';
            return 0;
        }
        if (monitor_options.show_help_info) {
            cout << make_man_page(monitor_argv_parser, argv[0]) << '\n';
            return 0;
        }
        if (latency_options.show_help_info) {
            cout << make_man_page(latency_argv_parser, argv[0]) << '\n';
            return 0;
        }
        if (list_show_help_info) {
            cout << make_man_page(list_argv_parser, argv[0]) << '\n';
            return 0;
        }

        hozon::netaos::log::LogLevel bag_tool_level = hozon::netaos::log::LogLevel::kError;
        if ("" != log_level) {
            if ("kDebug" == log_level) {
                bag_tool_level = hozon::netaos::log::LogLevel::kDebug;
                echo_options.open_proto_log = true;
                monitor_options.open_proto_log = true;
            } else if ("kInfo" == log_level) {
                bag_tool_level = hozon::netaos::log::LogLevel::kInfo;
            } else if ("kWarn" == log_level) {
                bag_tool_level = hozon::netaos::log::LogLevel::kWarn;
            } else if ("kError" == log_level) {
                bag_tool_level = hozon::netaos::log::LogLevel::kError;
            }
            // hozon::netaos::data_tool_common::DataToolsLogger::GetInstance().GetTopicLogger()->SetLogLevel(bag_tool_level);
            // hozon::netaos::data_tool_common::DataToolsLogger::GetInstance().GetCommonLogger()->SetLogLevel(bag_tool_level);
        }
        hozon::netaos::data_tool_common::DataToolsLogger::GetInstance().InitLogg(bag_tool_level);

        if (echo_options.topics.size() > 0) {
            type = 1;
            echo = new Echo();
            echo->Start(echo_options);
            echo = nullptr;
            delete echo;
        } else if (hz_options.events.size() > 0 || hz_options.monitor_all) {
            setTerminalInputMode(false);
            type = 2;
            hz = new Hz();
            hz->Start(hz_options);
            delete hz;
            hz = nullptr;
            setTerminalInputMode(true);

        } else if (monitor_options.events.size() > 0 || monitor_options.monitor_all) {
            type = 3;
            monitor = new Monitor();
            monitor->Start(monitor_options);
            delete monitor;
            monitor = nullptr;
        } else if (latency_options.topics.size() > 0) {
            type = 4;
            latency = new Latency();
            latency->Start(latency_options);
            delete latency;
            latency = nullptr;
        } else if (list_execute) {
            List list;
            list.Start(list_options);
        }

    } else {
        cout << make_man_page(hztool_argv_parser, argv[0]) << '\n';
    }

    return 0;
}