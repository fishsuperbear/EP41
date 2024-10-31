#include <iostream>
#include <getopt.h>
#include <thread>
#include <unistd.h>
#include <csignal>
#include <map>
#include "adf-lite/service/aldbg/utility/utility.h"
#include "adf-lite/include/dbg_info.h"
#include "adf-lite/include/adf_lite_internal_logger.h"
#include "adf-lite/include/sig_stop.h"
#include "adf-lite/include/cm_reader.h"
#include "cm/include/proto_cm_reader.h"
#include "proto/test/soc/dbg_msg.pb.h"
#include "cm/include/proto_method.h"

using namespace hozon::netaos::adf_lite;

uint32_t _show_counter = 20;
bool stop_flag = false;
enum class CmdType : int32_t {
    CMD_NONE = 0,
    CMD_HELP = 1,
    CMD_FREQ = 2,
    CMD_ECHO = 3,
    CMD_EXECUTOR = 4,
    CMD_TOPIC = 5,
    CMD_TOPICINFO = 6,
    CMD_SHOWALL = 7,
    CMD_RECORD = 8,
    CMD_PLAY = 9
};

CmdType type = CmdType::CMD_NONE;
std::string para;
std::string para2;
bool record_status;
void Help(){
	std::cout<<R"deli(
**********************************************************************************************************************************
**  Usage: aldbg [OPTION...] [SECTION]                                                                                          **
**  aldbg -h                                                                Show help info.                                     **
**  aldbg freq                                                              Show frequency.                                     **
**  aldbg executors Lite1                                                   Show executors managed by adf-lite.                 **
**  aldbg topics Lite1 FisheyePerception                                    Show topic's data.                                  **
**  aldbg record /lite/Lite1/event/workresult1 true                         Start/stop record topics.                           **
**  aldbg play /lite/Lite1/event/workresult1 true                           Start/stop play topics..                            **
**********************************************************************************************************************************)deli"<<std::endl;
}

void PrintFreq(std::shared_ptr<hozon::adf::lite::dbg::FreqDebugMessage> freq_msg) {
    ADF_EARLY_LOG << std::setw(36) << "topic"
            << std::setw(16) << "rate"
            << "  " << std::setw(16) << "min_delta"
            << "  " << std::setw(16) << "max_delta"
            << "  " << std::setw(16) << "std_dev";
    ADF_EARLY_LOG << "====================================================================================================================";

        for (int i = 0; i < freq_msg->mutable_elements()->size(); ++i) {
            auto ele = freq_msg->mutable_elements()->at(i);

            ADF_EARLY_LOG << std::fixed << std::setprecision(6)
                << std::setw(36) << ele.topic()
                << std::setw(16) << ele.freq()
                << std::setw(16) << ele.min_delta_us() / 1000
                << std::setw(16) << ele.max_delta_us() / 1000
                << std::setw(16) << ele.std_dev_us() / 1000;
        }
    ADF_EARLY_LOG << " ";
}

void FreqProc() {
    hozon::netaos::cm::ProtoCMReader<hozon::adf::lite::dbg::FreqDebugMessage> reader;

    int32_t ret = reader.Init(0, "/lite/Lite1/info", PrintFreq);
    if (ret < 0) {
        ADF_EARLY_LOG << "reader init failed!";
    }
    sleep(20);
    reader.Deinit();
}

void PrintContent(const std::string& content) {
    hozon::adf::lite::dbg::EchoDebugMessage msg;
    msg.ParseFromString(content);

    for (int i = 0; i < msg.mutable_elements()->size(); ++i) {
        ADF_EARLY_LOG << msg.mutable_elements()->at(i).topic() << " : " << msg.mutable_elements()->at(i).content();
    }

    ADF_EARLY_LOG << "-----------------------\n";
}

void ReceiveTopic(const std::string& topic, std::shared_ptr<CmProtoBuf> data) {
    ADF_EARLY_LOG << "topic_type : " << data->name();
    std::string msg(data->str().data(), data->str().size());

    ADF_EARLY_LOG << "Serialized msg val: " << msg;
}

void RequestCommand(const std::string& cmd, const std::vector<std::string>& topics, const bool status)
{
    std::map<std::string, std::vector<std::string>> topics_map;
    // 将topics按process进行分类
    for (auto & topic: topics) {
        std::string process_name;
        int32_t res = GetProcessFromCmTopic(topic, process_name);
        if (res == -1) {
            continue;
        }
        ADF_EARLY_LOG << "topic :" << topic << " process_name:" << process_name;
        if (process_name.size() > 0) {
            topics_map[process_name].push_back(topic);
        }
    }

    hozon::netaos::cm::ProtoMethodClient<hozon::adf::lite::dbg::EchoRequest, hozon::adf::lite::dbg::GeneralResponse> _proto_client;

    for (auto iter = topics_map.begin(); iter != topics_map.end(); ++iter) {
        int32_t ret = _proto_client.Init(0, "/lite/" + iter->first + "/method/cmd");
        std::shared_ptr<hozon::adf::lite::dbg::EchoRequest> req(new hozon::adf::lite::dbg::EchoRequest);
        req->set_cmd(cmd);
        req->set_status(status);
        for (auto &topic: iter->second) {
            hozon::adf::lite::dbg::EchoRequest::Element *ele = req->add_elements();
            ele->set_topic(topic);
        }

        std::shared_ptr<hozon::adf::lite::dbg::GeneralResponse> resp(new hozon::adf::lite::dbg::GeneralResponse);
        _proto_client.WaitServiceOnline(500);
        ret = _proto_client.Request(req, resp, 1000);
        _proto_client.DeInit();
        if (ret < 0) {
            ADF_EARLY_LOG << "Cmd: " << req->cmd() << "; Fail to request, ret " << ret;
        } else {
            ADF_EARLY_LOG << "Cmd: " << req->cmd() << " Resp= " << resp->res();
        }
    }

    if ((cmd == "echo" || cmd == "record") && status && topics.size() > 0) {
        hozon::netaos::adf_lite::CMReader reader;
        reader.Init(0, topics.at(0), ReceiveTopic);
        sleep(6);
        reader.Deinit();
    }
}

void Echo(const std::vector<std::string>& topics, const bool status)
{
    std::string cmd("echo");
    RequestCommand(cmd, topics, status);
}

void Record(const std::vector<std::string>& topics, const bool status)
{
    std::string cmd("record");
    RequestCommand(cmd, topics, status);
}

void Play(const std::vector<std::string>& topics, const bool status)
{
    std::string cmd("play");
    RequestCommand(cmd, topics, status);
}

#define WD(statement) std::setw(16) << std::left << statement
#define WIDTH(statement, width) std::setw(width) << std::left << statement

std::string GetSysDate(void) {
    // get current time
    auto now = std::chrono::system_clock::now();

    // get number of milliseconds for the current second
    // (remainder after division into seconds)
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    // convert to std::time_t in order to convert to std::tm (broken time)
    auto timer = std::chrono::system_clock::to_time_t(now);

    // convert to broken time
    std::tm bt = *std::localtime(&timer);

    std::ostringstream oss;

    oss << std::put_time(&bt, "%H:%M:%S"); // HH:MM:SS
    oss << '.' << std::setfill('0') << std::setw(3) << ms.count();

    return oss.str();
}

void ShowLatencyInfo(std::shared_ptr<hozon::adf::lite::dbg::LatencyInfo> msg) {

    if (_show_counter == 20) {
        _show_counter = 0;
        std::cout << WD(" ");
        std::cout << WD(" ");
        for (auto link_name : msg->link_names()) {
            std::cout << WD(link_name);
        }
        std::cout << std::endl;
    }
    _show_counter++;
    if (msg->after_process()) {
        std::cout << "\033[0;31m" << WD(GetSysDate());
        std::cout << WIDTH(msg->instance_name() + "(after)", 45);

        for (double latency : msg->latencies()) {
            std::cout << WD(latency);
        }
        std::cout << "\033[0m" << std::endl;
    } else {
        std::cout << "\033[0;34m" << WD(GetSysDate());
        std::cout << WIDTH(msg->instance_name() + "(befor)", 45);

        for (double latency : msg->latencies()) {
            std::cout << WD(latency);
        }
        std::cout << "\033[0m" << std::endl;
    }
}

void LatencyServer() {
    hozon::netaos::cm::ProtoCMReader<hozon::adf::lite::dbg::LatencyInfo> reader;

    int32_t ret = reader.Init(0, "latency_information", ShowLatencyInfo);
    if (ret < 0) {
        ADF_EARLY_LOG << "reader init failed!";
    }
    ADF_EARLY_LOG << "reader init success!";
    while (!stop_flag) {
        sleep(1);
    }
    // sleep(20);
    reader.Deinit();
}

void ShowExecutors(const std::string lite_process) {
    hozon::netaos::cm::ProtoMethodClient<hozon::adf::lite::dbg::EchoRequest, hozon::adf::lite::dbg::GeneralResponse> _proto_client;

    int32_t ret = _proto_client.Init(0, "/lite/" + lite_process + "/method/cmd");
    std::shared_ptr<hozon::adf::lite::dbg::EchoRequest> req(new hozon::adf::lite::dbg::EchoRequest);
    req->set_cmd("executors");
    std::shared_ptr<hozon::adf::lite::dbg::GeneralResponse> resp(new hozon::adf::lite::dbg::GeneralResponse);
    _proto_client.WaitServiceOnline(500);
    ret = _proto_client.Request(req, resp, 1000);
    _proto_client.DeInit();
    if (ret < 0) {
        ADF_EARLY_LOG << "Cmd: " << req->cmd() << "; Fail to request, ret " << ret;
    } else {
        ADF_EARLY_LOG << std::left << "Adf-lite managed executors is as follows:";
        char title[50];
        sprintf(title, "%4s%24s", "NO.", "executor");
        ADF_EARLY_LOG << std::right << WITH_LINE(title);
        int i = 0;
        for (auto it = resp->data().begin(); it != resp->data().end(); i++, it++) {
            ADF_EARLY_LOG << std::right << std::setw(4) << i + 1 << std::setw(24) << *it;
        }
    }
}

void ShowExecutorTopics(const std::string lite_process, const std::string executor_name) {
    hozon::netaos::cm::ProtoMethodClient<hozon::adf::lite::dbg::EchoRequest, hozon::adf::lite::dbg::GeneralResponse> _proto_client;

    int32_t ret = _proto_client.Init(0, "/lite/" + lite_process + "/method/cmd");
    std::shared_ptr<hozon::adf::lite::dbg::EchoRequest> req(new hozon::adf::lite::dbg::EchoRequest);
    req->set_cmd("topics");
    req->set_other(executor_name);
    std::shared_ptr<hozon::adf::lite::dbg::GeneralResponse> resp(new hozon::adf::lite::dbg::GeneralResponse);
    _proto_client.WaitServiceOnline(500);
    ret = _proto_client.Request(req, resp, 1000);
    _proto_client.DeInit();
    if (ret < 0) {
        ADF_EARLY_LOG << "Cmd: " << req->cmd() << "; Fail to request, ret " << ret;
    } else {
        ADF_EARLY_LOG << std::left << "The executor " << executor_name << " received topics is as follows:";
        char title[50];
        sprintf(title, "%4s%24s", "NO.", "topic");
        ADF_EARLY_LOG << std::right << WITH_LINE(title);
        int i = 0;
        for (auto it = resp->data().begin(); it != resp->data().end(); i++, it++) {
            ADF_EARLY_LOG << std::right << std::setw(4) << i + 1 << std::setw(24) << *it;
        }
    }
}

void signalHandler(int signum)
{
    ADF_EARLY_LOG << "signalHandler";
    if (type == CmdType::CMD_ECHO || type == CmdType::CMD_RECORD) {
        ADF_EARLY_LOG << "stop record";
        Record({para}, false);
    }
}


int main(int argc, char* argv[]) {
    /* 抓取信号量。*/
    signal(SIGTERM, signalHandler);
    signal(SIGINT, signalHandler);

    hozon::netaos::log::InitLogging("Aldbg","Aldbg",hozon::netaos::log::LogLevel::kInfo,
			 hozon::netaos::log::HZ_LOG2FILE, "./", 10, 20);
    hozon::netaos::adf_lite::SigHandler::GetInstance().Init();

    bool need_block = false;
    std::shared_ptr<std::thread> _thread;
    switch (argc)
	{
	case 0:
    case 1: {
        Help();
        exit(0);
    }
    break;
    case 2: {
		if (strcmp("freq", argv[1]) == 0) {
            need_block = false;
            _thread = std::make_shared<std::thread>(&FreqProc);
        } else if (strcmp("latency", argv[1]) == 0) {
            need_block = true;
            _thread = std::make_shared<std::thread>(&LatencyServer);
        } else {
            Help();
            exit(0);
        }
    }
		break;
	case 3:{
        if (strcmp("executors", argv[1]) == 0) {
            _thread = std::make_shared<std::thread>(&ShowExecutors, std::string(argv[2]));
		} else {
            Help();
            exit(0);
		}

	}
        break;
	case 4:{
        if (strcmp("topics",argv[1]) == 0){
            _thread = std::make_shared<std::thread>(&ShowExecutorTopics, std::string(argv[2]), std::string(argv[3]));

		} else if (strcmp("echo", argv[1]) == 0) {
            std::vector<std::string> processes;
            processes.push_back(std::string(argv[2]));
            _thread = std::make_shared<std::thread>(&Echo, processes, (strcmp("true", argv[3]) == 0));
		} else if (strcmp("record", argv[1]) == 0) {
            std::vector<std::string> processes;
            processes.push_back(std::string(argv[2]));
            _thread = std::make_shared<std::thread>(&Record, processes, (strcmp("true", argv[3]) == 0));
		} else if (strcmp("play", argv[1]) == 0) {
            std::vector<std::string> processes;
            processes.push_back(std::string(argv[2]));
            _thread = std::make_shared<std::thread>(&Play, processes, (strcmp("true", argv[3]) == 0));
		} else {
            Help();
            exit(0);
		}
	}
        break;

	default:
        Help();
        exit(0);
		break;
	}

    if (need_block) {
        hozon::netaos::adf_lite::SigHandler::GetInstance().NeedStopBlocking();
    }
    stop_flag = true;
    if (_thread->joinable()) {
        _thread->join();
    }
}
