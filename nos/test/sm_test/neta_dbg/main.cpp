#include <thread>
#include <chrono>
#include <cstdint>
#include <signal.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <map>
#include <vector>
#include "log/include/logging.h"
#include "em/include/proctypes.h"
#include "sm/include/state_client.h"

using namespace std;
using namespace hozon::netaos::sm;
using namespace hozon::netaos::em;
using namespace hozon::netaos::log;

class NetaDbg {
public:
	NetaDbg(){};
	~NetaDbg(){};
	void PrintProcessStatus();
	void PrintHelp();
	void SwitchMode(std::string new_mode);
	void ShowCurrMode();

private:
	StateClient s_client;
};

void NetaDbg::PrintProcessStatus()
{
	vector<ProcessInfo> processStatus;
	s_client.GetProcessInfo(processStatus);

	vector<std::string> process_state = {"IDLE", "STARTING", "RUNNING", "TERMINATING", "TERMINATED", "ABORTED"};

	printf("\n\033[4m%-30s:%15s\033[0m\n", "ProcessName", "ProcessState");
	for(auto it=processStatus.begin(); it !=processStatus.end(); it++) {
		if (it->procstate == ProcessState::ABORTED) {
			printf("%-30s:\033[31m%15s\033[0m\n", it->procname.c_str(), process_state[(int)it->procstate].c_str());
		} else if (it->procstate == ProcessState::TERMINATED || it->procstate == ProcessState::TERMINATING) {
			printf("%-30s:\033[33m%15s\033[0m\n", it->procname.c_str(), process_state[(int)it->procstate].c_str());
		} else {
			printf("%-30s:%15s\n", it->procname.c_str(), process_state[(int)it->procstate].c_str());
		}
	}
	printf("\n");
}

void NetaDbg::PrintHelp()
{
	printf("use like this\n");
	printf("neta_dbg -q\n");//当命令行选项字符不包括在optstring里或者选项缺少必要参数时
}

void NetaDbg::SwitchMode(std::string new_mode)
{
	printf("SwitchMode to %s\n", new_mode.c_str());
	int32_t res = s_client.SwitchMode(new_mode);

	if (res == 0) {
		printf("SwitchMode to %s success!\n", new_mode.c_str());
	} else {
		printf("SwitchMode to %s failed!\n", new_mode.c_str());
	}
}

void NetaDbg::ShowCurrMode()
{
	std::string curr_mode;
	cout << "ShowCurrMode" << endl;
	int32_t res = s_client.GetCurrMode(curr_mode);

	if (res == 0) {
		printf("Current Mode is %s\n", curr_mode.c_str());
	} else {
		printf("ShowCurrMode failed!\n");
	}
}

int main(int argc, char*argv[])
{
	int result;
	opterr=0;//不输出错误信息

	hozon::netaos::log::InitLogging("neta_dbg","neta_dbg",hozon::netaos::log::LogLevel::kOff,
			hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE, "./", 10, 20);
	NetaDbg netadbg;
	while((result=getopt(argc,argv,"s:qg::")) !=-1)
	{
		switch(result)
		{
			case 'q':
				netadbg.PrintProcessStatus();
			break;
			case 's':
				netadbg.SwitchMode(optarg);
			break;
			case 'g':
				netadbg.ShowCurrMode();
			break;
			case '?':
				netadbg.PrintHelp();
			break;
			default:
				printf("default,result=%c\n",result);
			break;
		}
	}
	printf("neta_dbg over --------------------------------------------\n");
	return 0;
}
