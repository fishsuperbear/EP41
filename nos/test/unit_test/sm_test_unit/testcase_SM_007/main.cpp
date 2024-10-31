#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <cstring>
#include <signal.h>
#include <unistd.h>
#include "em/include/proctypes.h"
#include "em/include/exec_client.h"
#include "em/include/logger.h"
#include "main.h"
#include <memory>
#include "gtest/gtest.h"
#include "log/include/logging.h"

/***************测试说明***************/
/*
测试场景：EM拉起进程sm_proc_b, sm_proc_b中注册了前处理，后处理函数。切换模式时，验证注册的函数是否被调用到。
拉起的进程，A模式为[Normal:1, Parking:3, OTA:1]
拉起的进程，B模式为[Driving:1, OTA:2]               Normal:1,Parking:2,使用sed删除
拉起的进程，sm_proc_b模式为[Driving.99，Parking.3，Normal.99] 
期望结果: EM拉起进程sm_proc_b时，前处理，后处理函数注册成功。
*/
/***************测试说明***************/
std::string id;

using namespace hozon::netaos::em;

#define EM_WAIT_TIME 10
class StateClientTest_HasServer:public ::testing::Test{
protected:
	static void SetUpTestSuite() {
		cout << "=========SetUpTestSuite=========" << endl;
	}
	static void TearDownTestSuite() {
		cout << "=========TearDownTestSuite=========" << endl;
	}
	void SetUp() override {
		
		instance = make_unique<StateClientImpl>();
	}

	void TearDown() override {
		cout << "=========TearDown=========" << endl;
		int res = system("chmod +x ../scripts/TearDown.sh; ../scripts/TearDown.sh");
		cout << res << endl;
		sleep(EM_WAIT_TIME);
	}
protected:
	std::unique_ptr<StateClientImpl> instance;
	pid_t m_pid;
public:
	static int32_t preProcess(const std::string& old_mode, const std::string& new_mode){
		cout << "do preProcess()!!!! id is " << id <<endl;
		postProcess(old_mode, new_mode, true);
		
		return 0;
	}
	static void postProcess(const std::string& old_mode, const std::string& new_mode, const bool succ) {
		if (succ) {
			cout << "do postProcess()!!!! succ is true" << id <<endl;

		} else {
			cout << "do postProcess()!!!! succ is false" << id <<endl;

		}
	}
};

bool find_process_info(vector<ProcessInfo> &process_info, ProcessInfo& target)
{
	for (auto &process: process_info) {
		if (process.group == target.group && process.procname == target.procname && process.procstate == target.procstate) {
			return true;
		}
	}
	return false;
}

void Check(std::unique_ptr<StateClientImpl> &instance, string expect_mode, vector<ProcessInfo> &expect_pi)
{
	//判断
	std::string curr_mod;
	int32_t res = instance->GetCurrMode(curr_mod);  // 8004
	EXPECT_EQ(res, 0);
	cout << "============================= check start =============================" << endl;
	cout << "GetCurrMode result is " << curr_mod << endl;
	EXPECT_EQ(curr_mod, expect_mode);

	vector<ProcessInfo> process_info;
	res = instance->GetProcessInfo(process_info);   // 8101
	EXPECT_EQ(res, 0);
	int process_info_size = process_info.size();
	EXPECT_EQ(process_info_size, static_cast<int>(expect_pi.size()));

	for(auto process :process_info) {
		cout << process.group << " " << process.procname << " " << static_cast<int>(process.procstate) << endl;
	}
	if (process_info_size >= static_cast<int>(expect_pi.size())) {
		//判断 expect_pi中的每一个在process_info中都有
		for (int i = 0; i < static_cast<int>(expect_pi.size()); i++) {

			bool find_res = find_process_info(process_info, expect_pi[i]);
			if (find_res == false) {
				cout << "expect process info not found:[" << expect_pi[i].group << " " << expect_pi[i].procname << " " << static_cast<int>(expect_pi[i].procstate) << "]" << endl;
			}
			EXPECT_EQ(find_res, true);
		}
	}
	cout << "============================= check end =============================" << endl;
	
}

void executeCMD(const char *cmd, char *result)   
{   
    char buf_ps[1024];   
    char ps[1024]={0};   
    FILE *ptr;   
    strcpy(ps, cmd);   
    if((ptr=popen(ps, "r"))!=NULL)   
    {   
        while(fgets(buf_ps, 1024, ptr)!=NULL)   
        {   
           strcat(result, buf_ps);   
           if(strlen(result)>1024)   
               break;   
        }   
        pclose(ptr);   
        ptr = NULL;   
    }   
    else  
    {   
        printf("popen %s error", ps);   
    }   
}

void CheckLog(string keyword, int32_t count)
{
	char hdmi_result[1024];
	char cmd[1024];
	sprintf(cmd, "chmod +x ../scripts/check_log.sh; ../scripts/check_log.sh '%s' %d", keyword.c_str(), count);
	cout << cmd << endl;
	
	executeCMD(cmd, hdmi_result);
	cout << hdmi_result << endl;
	EXPECT_EQ(hdmi_result[strlen(hdmi_result) - 2], '0');
}

TEST_F(StateClientTest_HasServer, SwitchMode_HasServer) {
	{
		//将需要由em拉起的进程拷贝到sm_test/emproc目录中
		int res = system("chmod +x ../scripts/SetUp.sh; ../scripts/SetUp.sh");
		cout << res << endl;
		sleep(EM_WAIT_TIME * 2);
		vector<ProcessInfo> expect{{1, "hz_app_aProcess", ProcessState::RUNNING}, {0, "hz_app_bProcess", ProcessState::IDLE}, {99,"hz_sm_bProcess", ProcessState::RUNNING}};
		Check(instance, "Normal", expect);

		//检查proB_log.txt文件中包含注册成功的日志
		CheckLog("SMProcB", 12);
	}
	
	{
		cout << "1.SwitchMode to Driving" << endl;
		int32_t res3 = instance->SwitchMode("Driving");
		EXPECT_EQ(res3, 0);
		//判断
		vector<ProcessInfo> expect{{1, "hz_app_bProcess", ProcessState::RUNNING},{1,"hz_app_bProcess", ProcessState::RUNNING},{99,"hz_sm_bProcess", ProcessState::RUNNING}};
		Check(instance, "Driving", expect);
	}
	
	{
		//判断=> "Normal" 模式切换成功
		cout << "2.SwitchMode to Normal" << endl;
		int32_t res3 = instance->SwitchMode("Normal");
		EXPECT_EQ(res3, 0);

		// //判断
		vector<ProcessInfo> expect{{1, "hz_app_aProcess", ProcessState::RUNNING},{0,"hz_app_bProcess", ProcessState::TERMINATED},{99,"hz_sm_bProcess", ProcessState::RUNNING}};
		Check(instance, "Normal", expect);
	}

	{
		//判断=> "Normal" 模式切换成功
		cout << "3.SwitchMode to Normal" << endl;
		int32_t res3 = instance->SwitchMode("Normal");
		EXPECT_EQ(res3, 0);

		//判断
		vector<ProcessInfo> expect{{1, "hz_app_aProcess", ProcessState::RUNNING},{0,"hz_app_bProcess", ProcessState::TERMINATED},{99,"hz_sm_bProcess", ProcessState::RUNNING}};
		Check(instance, "Normal", expect);
	}

	{
		//判断=> "Parking" 模式切换成功
		cout << "4.SwitchMode to Parking" << endl;
		int32_t res3 = instance->SwitchMode("Parking");
		EXPECT_EQ(res3, 0);

		//判断
		vector<ProcessInfo> expect{{3, "hz_app_aProcess", ProcessState::RUNNING},{2,"hz_app_bProcess", ProcessState::RUNNING},{0,"hz_sm_bProcess", ProcessState::TERMINATED}};
		Check(instance, "Parking", expect);
	}

	{
		//判断=> "OTA" 模式切换成功
		cout << "5.SwitchMode to OTA" << endl;
		int32_t res3 = instance->SwitchMode("OTA");
		EXPECT_EQ(res3, 0);

		//判断
		vector<ProcessInfo> expect{{1, "hz_app_aProcess", ProcessState::RUNNING},{2,"hz_app_bProcess", ProcessState::RUNNING},{0,"hz_sm_bProcess", ProcessState::TERMINATED}};
		Check(instance, "OTA", expect);
	}

	{
		//判断=> "Normal" 模式切换成功
		cout << "6.SwitchMode to Normal" << endl;
		int32_t res3 = instance->SwitchMode("Normal");
		EXPECT_EQ(res3, 0);

		//判断
		vector<ProcessInfo> expect{{1, "hz_app_aProcess", ProcessState::RUNNING},{0,"hz_app_bProcess", ProcessState::TERMINATED},{99,"hz_sm_bProcess", ProcessState::RUNNING}};
		Check(instance, "Normal", expect);
	}
}

int main(int argc, char* argv[])
{
	hozon::netaos::log::InitLogging("sm_ut7","sm_ut7",hozon::netaos::log::LogLevel::kTrace,
			hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE, "./", 10, 20);
	hozon::netaos::log::CreateLogger("-C", "", hozon::netaos::log::LogLevel::kInfo);
	testing::InitGoogleTest(&argc,argv);
    int res = RUN_ALL_TESTS();
	if (res == 0) {
		cout << res <<  "!!!!!hz_test_success!!!!!" << endl;
	}
	else {
		cout << res <<  "!!!!!hz_test_fail!!!!!" << endl;
	}
}
