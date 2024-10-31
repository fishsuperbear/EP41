#include "neta_dbg_manager.h"
#include <time.h>

namespace hozon {
namespace netaos {
namespace dbg {

NetaDbgManager::NetaDbgManager(){
	m_st_cli = std::make_shared<StateClientZmq>();
}

NetaDbgManager::~NetaDbgManager(){}

void NetaDbgManager::DeInit(){
    this->Sysdate();
}

int32_t NetaDbgManager::Init(){
	this->Sysdate();
    return 0;
}

void NetaDbgManager::Help(){
	std::cout<<R"deli(
    usage:  smdbg
            smdbg -h                                  Show help info.
            smdbg restart < processName >             Restart process.
            smdbg query { modeList | processStatus }  Query mode list or proc status.
            smdbg request mode < modeName >           Switch to mode.
            smdbg set startupMode < modeName >        Modify default mode.
            smdbg list                                Show all modes and all proc.
            smdbg reboot                              SoC reboot.
            smdbg reset                               Orin board reset.
)deli"<<std::endl;
}

void NetaDbgManager::GetCurrentMode()
{
	std::string curr_mode;
	int32_t res = m_st_cli->GetCurrMode(curr_mode);
	if (res == 0) {
		printf("Current Mode is %s\n", curr_mode.c_str());
	} else {
		printf("\033[0m\033[1;31m Get cur mode failed\033[0m\n");
	}
}

void NetaDbgManager::GetModeList()
{
	std::vector<std::string> vect;
	m_st_cli->GetModeList(vect);

    std::string curr_mode;
	m_st_cli->GetCurrMode(curr_mode);

	int index = 0;
	printf("\n\033[1;4m%3s    %20s\033[0m\n", "No", "Mode");
	for(auto it : vect){//vect.begin(); it !=vect.end(); it++) {
		index++;
		if (it == curr_mode) {
			printf("\033[32m%3d    %20s\033[0m\n", index, it.c_str());
		} else {
			printf("\033[0m%3d    %20s\033[0m\n", index, it.c_str());
		}
	}
}

void NetaDbgManager::GetModeListDetailInfo()
{
    std::vector<hozon::netaos::zmqipc::process_info> process_list;
    m_st_cli->GetModeListDetailInfo(process_list);

    std::vector<std::pair<std::string, std::string>> color_list = {
        {"\033[32m", "\033[0m"},
        {"\033[33m", "\033[0m"},
        {"\033[34m", "\033[0m"},
        {"\033[35m", "\033[0m"},
        {"\033[36m", "\033[0m"},
        {"\033[37m", "\033[0m"},
        {"\033[38m", "\033[0m"},
        {"\033[39m", "\033[0m"},
        {"\033[40m", "\033[0m"},
        {"\033[41m", "\033[0m"}
    };

    printf("\n\033[1;4m%3s  %40s\033[0m\n", "Mode", "ProcessName");
    std::string last_mode;
    int index = -1;
    int color_num = color_list.size();
    for(const auto &process : process_list){
        if (last_mode != process.mode_name()) {
            ++index;
            index = index % color_num;
            if (!last_mode.empty()) {
                printf("\n");
            }
            printf("%s%3s    %40s%s\n", color_list[index].first.c_str(), process.mode_name().c_str(), "", color_list[index].second.c_str());
            last_mode = process.mode_name();
        }
        printf("%s%3s    %40s%s\n", "", color_list[index].first.c_str(), process.procname().c_str(), color_list[index].second.c_str());
    }
    printf("\n");
}

void NetaDbgManager::GetProcessState()
{
	std::vector<ProcessInfo> vect;
	m_st_cli->GetProcessInfo(vect);

    std::string curr_mode;
	m_st_cli->GetCurrMode(curr_mode); 

	vector<std::string> process_state = {"IDLE", "STARTING", "RUNNING", "TERMINATING", "TERMINATED", "ABORTED"};

	printf("\n\033[0m\033[1;4m%10s:%35s:%20s:%15s:\033[0m\n", "CurrentMode", "ProcessName", "ProcessState","Priority");
    if(curr_mode == "Abnormal"){
        printf("\n\033[0m\033[1;31m%10s\033[0m\n", curr_mode.c_str());
    }else{
        printf("\n%10s\n", curr_mode.c_str());
	}

	size_t n = vect.size();
	ProcessInfo arry[n];
    for (size_t i = 0; i < n; i++){
        arry[i] = vect[i];
    }
	std::sort(arry ,arry + n,Sortcmp);

    for (size_t j = 0; j < n; j++){
		ProcessInfo info = arry[j];
		stringstream ss;
        ss<<info.group;
		if (info.procstate == ProcessState::ABORTED) {
			printf("\033[0m%10s%35s\033[31m%20s\033[0m%15s\n", "",info.procname.c_str(), process_state[(int)info.procstate].c_str(),(info.group == 0 ? "-": ss.str().c_str()));
		} else if (info.procstate == ProcessState::IDLE){
			// printf("\033[0m%10s%35s\033[32m%20s\033[0m%15s\n", "",info.procname.c_str(), process_state[(int)info.procstate].c_str(),(info.group == 0 ? "-": ss.str().c_str()));
			printf("\033[0m%10s%35s%20s%15s\n", "",info.procname.c_str(), process_state[(int)info.procstate].c_str(),(info.group == 0 ? "-": ss.str().c_str()));
		} else if (info.procstate == ProcessState::TERMINATED || info.procstate == ProcessState::TERMINATING) {
			printf("\033[0m%10s%35s\033[33m%20s\033[0m%15s\n", "",info.procname.c_str(), process_state[(int)info.procstate].c_str(),(info.group == 0 ? "-": ss.str().c_str()));
		} else if (info.procstate == ProcessState::STARTING) {
            printf("\033[0m%10s%35s\033[36m%20s\033[0m%15s\n", "",info.procname.c_str(), process_state[(int)info.procstate].c_str(),(info.group == 0 ? "-": ss.str().c_str()));
		} else {
			// printf("\033[0m%10s%35s%20s%15s\n", "",info.procname.c_str(), process_state[(int)info.procstate].c_str(),(info.group == 0 ? "-": ss.str().c_str()));
			printf("\033[0m%10s%35s\033[32m%20s\033[0m%15s\n", "",info.procname.c_str(), process_state[(int)info.procstate].c_str(),(info.group == 0 ? "-": ss.str().c_str()));
		}
    }
}


void NetaDbgManager::SwitchMode(std::string & mode)
{
	std::string title_str = "SwitchMode to " + mode;
	printf("\n\033[0m\033[1;36m%s\033[0m\n", title_str.c_str());
	int32_t res = m_st_cli->SwitchMode(mode);
	if (res == 0) {
		std::string succ_str = title_str + " success.";
		printf("\033[0m\033[1;32m%s\033[0m\n", succ_str.c_str());
	} else {
		std::string fail_str = title_str + " failed.";
		printf("\033[0m\033[1;31m%s\033[0m\n", fail_str.c_str());
	}
}

void NetaDbgManager::RestartProcess(std::string & proc)
{
	std::string title_str = "Restart " + proc;
	printf("\n\033[0m\033[1;36m%s\033[0m\n", title_str.c_str());
	int32_t res = m_st_cli->ProcRestart(proc);
	if (res == 0) {
		std::string succ_str = title_str + " success.";
		printf("\033[0m\033[1;32m%s\033[0m\n", succ_str.c_str());
	} else {
		std::string fail_str = title_str + " failed.";
		printf("\033[0m\033[1;31m%s\033[0m\n", fail_str.c_str());
	}
}

void NetaDbgManager::StopMode(){
    std::string title_str = "Stop cur mode";
	printf("\n\033[0m\033[1;36m%s\033[0m\n", title_str.c_str());
	int32_t res = m_st_cli->StopMode();
	if (res == 0) {
		std::string succ_str = title_str + " success.";
		printf("\033[0m\033[1;32m%s\033[0m\n", succ_str.c_str());
	} else {
		std::string fail_str = title_str + " failed.";
		printf("\033[0m\033[1;31m%s\033[0m\n", fail_str.c_str());
	}
}


void NetaDbgManager::SetDefaultStartupMode(std::string & mode){
	std::string title_str = "Set default startup mode " + mode;
	printf("\n\033[0m\033[1;36m%s\033[0m\n", title_str.c_str());
	int32_t res = m_st_cli->SetDefaultMode(mode);
	if (res == 0) {
		std::string succ_str = title_str + " success.";
		printf("\033[0m\033[1;32m%s\033[0m\n", succ_str.c_str());
	} else {
		std::string fail_str = title_str + " failed.";
		printf("\033[0m\033[1;31m%s\033[0m\n", fail_str.c_str());
	}
}

int32_t NetaDbgManager::Reboot(){
    auto client_zmq = std::make_shared<hozon::netaos::zmqipc::ZmqIpcClient>();
    client_zmq->Init(SSM_ZMQ_END_POINT);
    std::string reply;
    auto res = client_zmq->Request("reboot_soc", reply, 5000);
    if (res == 0 && reply == "success") {
        printf("\033[0m\033[1;32mrequest mcu reboot succ\033[0m\n");
    } else {
		printf("request mcu reboot failed, select nv trigger sys reboot\n");
        int res = system((const char *)TEGRA_TRIGGER_SYS_REBOOT);
    }
    client_zmq->Deinit();
	return res;
}

int32_t NetaDbgManager::Reset(){
    auto client_zmq = std::make_shared<hozon::netaos::zmqipc::ZmqIpcClient>();
    client_zmq->Init(SSM_ZMQ_END_POINT);
    std::string reply;
    auto res = client_zmq->Request("reboot_orin", reply, 5000);
    if (res == 0 && reply == "success") {
        printf("\033[0m\033[1;32mrequest mcu reset succ\033[0m\n");
    } else {
		printf("\033[0m\033[1;31mrequest mcu reset failed\033[0m\n");
	}
    client_zmq->Deinit();
	return res;
}

void NetaDbgManager::Sysdate() {
	time_t tim;
    tim = time(NULL);
	printf("%s", ctime(&tim));
}

uint32_t NetaDbgManager::CombResult(size_t n,size_t m){
    size_t sum=0, num=1;
    for(size_t i=1;i<=m;i++){
        num=num*n/i; n--;
        sum+=num;
    }
    return sum;
}

uint32_t NetaDbgManager::CombChars(size_t n,size_t m){
    size_t res=1;
    for(size_t i=1;i<=m;i++){
        res=res*(n-i+1)/i;
    }
    return res;
}

uint32_t NetaDbgManager::SumString(size_t len,string str){
    size_t sum=0 ,index=0;
    for(size_t i=0; i<len; i++){
        for(ssize_t j=index; j<str[i]-'a'; j++){
            sum+=CombChars(26-j-1,len-i-1);
        }
        index=str[i]-'a'+1;
    }
    return sum;
}

bool NetaDbgManager::Sortcmp(ProcessInfo &a,ProcessInfo &b){
	if(a.group == b.group){
		return SumString(1,a.procname) < SumString(1,b.procname);
        // return CombResult(26,a.procname.length()-1)+SumString(a.procname.length(),a.procname) < \
		//        CombResult(26,b.procname.length()-1)+SumString(b.procname.length(),b.procname);
	}else{
        return a.group < b.group;
	}
}

}}}
