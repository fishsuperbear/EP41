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
测试场景：测试存在EM服务端，存在StateServer的场景。EM拉起了a,b进程，主要确认SetDefaultMode有效。
拉起的进程，A模式为[Normal:1, Parking:3, OTA:1]
拉起的进程，B模式为[Driving:1, OTA:2]               Normal:1,Parking:2,使用sed删除

期望结果: 模式切换时，进程启动及关闭正确，主要确认SetDefaultMode有效。
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
	sleep(2);
	std::string curr_mod;
	int32_t res = instance->GetCurrMode(curr_mod);
	EXPECT_EQ(res, 0);
	cout << "============================= check start =============================" << endl;
	cout << "GetCurrMode result is " << curr_mod << endl;
	EXPECT_EQ(curr_mod, expect_mode);

	vector<ProcessInfo> process_info;
	res = instance->GetProcessInfo(process_info);
	EXPECT_EQ(res, 0);
	int process_info_size = process_info.size();
	EXPECT_EQ(process_info_size, static_cast<int>(expect_pi.size()));

	for(auto process :process_info) {
		cout << process.group << " " << process.procname << " " << static_cast<int>(process.procstate) << endl;
	}
	if (process_info_size == static_cast<int>(expect_pi.size())) {
		//判断 expect_pi中的每一个在process_info中都有
		for (int i = 0; i < process_info_size; i++) {

			bool find_res = find_process_info(process_info, expect_pi[i]);
			if (find_res == false) {
				cout << "expect process info not found:[" << expect_pi[i].group << " " << expect_pi[i].procname << " " << static_cast<int>(expect_pi[i].procstate) << "]" << endl;
			}
			EXPECT_EQ(find_res, true);
		}
	}
	cout << "============================= check end =============================" << endl;
	
}

TEST_F(StateClientTest_HasServer, SwitchMode_HasServer) {
	{
		//将需要由em拉起的进程拷贝到sm_test/emproc目录中
		int res = system("chmod +x ../scripts/SetUp.sh; ../scripts/SetUp.sh");
		cout << res << endl;
		sleep(EM_WAIT_TIME * 3);
		vector<ProcessInfo> expect{{1, "hz_app_aProcess", ProcessState::RUNNING},{0,"hz_app_bProcess", ProcessState::IDLE}};
		Check(instance, "Normal", expect);
	}
	
	{
		//修改DefaultMode为 Driving
		cout << "0.设置默认启动模式为Driving Mode" << endl;
		int32_t res = instance->SetDefaultMode("Driving");
		EXPECT_EQ(res, 0);

		cout << "1.SwitchMode to Driving" << endl;
		int32_t res3 = instance->SwitchMode("Driving");
		EXPECT_EQ(res3, 0);
		//判断
		vector<ProcessInfo> expect{{0, "hz_app_aProcess", ProcessState::TERMINATED},{1,"hz_app_bProcess", ProcessState::RUNNING}};
		Check(instance, "Driving", expect);
	}
	
	{
		//判断=> "Normal" 模式切换成功
		cout << "2.SwitchMode to Normal" << endl;
		int32_t res3 = instance->SwitchMode("Normal");
		EXPECT_EQ(res3, 0);

		//判断
		vector<ProcessInfo> expect{{1, "hz_app_aProcess", ProcessState::RUNNING},{0,"hz_app_bProcess", ProcessState::TERMINATED}};
		Check(instance, "Normal", expect);
	}

	{
		//结束EM进程，后重新拉起EM进程，预期进入Driving Mode，拉起了b
		cout << "2.结束EM进程后，重新拉起EM进程，预期进入Driving Mode" << endl;
		int res = system("chmod +x ../scripts/restart_em.sh; ../scripts/restart_em.sh");
		cout << res << endl;
		sleep(EM_WAIT_TIME * 3);
		vector<ProcessInfo> expect{{0, "hz_app_aProcess", ProcessState::IDLE},{1, "hz_app_bProcess", ProcessState::RUNNING}};
		Check(instance, "Driving", expect);
	}

	{
		//判断=> "Normal" 模式切换成功
		cout << "3.SwitchMode to Normal" << endl;
		int32_t res3 = instance->SwitchMode("Normal");
		EXPECT_EQ(res3, 0);

		//判断
		vector<ProcessInfo> expect{{1, "hz_app_aProcess", ProcessState::RUNNING},{0,"hz_app_bProcess", ProcessState::TERMINATED}};
		Check(instance, "Normal", expect);
	}


	{
		//判断=> "Parking" 模式切换成功
		cout << "4.SwitchMode to Parking" << endl;
		int32_t res3 = instance->SwitchMode("Parking");
		EXPECT_EQ(res3, 0);

		//判断
		vector<ProcessInfo> expect{{3, "hz_app_aProcess", ProcessState::RUNNING},{0,"hz_app_bProcess", ProcessState::TERMINATED}};
		Check(instance, "Parking", expect);
	}

	{
		//判断=> "OTA" 模式切换成功
		cout << "5.SwitchMode to OTA" << endl;
		int32_t res3 = instance->SwitchMode("OTA");
		EXPECT_EQ(res3, 0);

		//判断
		vector<ProcessInfo> expect{{1, "hz_app_aProcess", ProcessState::RUNNING},{2,"hz_app_bProcess", ProcessState::RUNNING}};
		Check(instance, "OTA", expect);
	}

	{
		//判断=> "Normal" 模式切换成功
		cout << "6.SwitchMode to Normal" << endl;
		int32_t res3 = instance->SwitchMode("Normal");
		EXPECT_EQ(res3, 0);

		//判断
		vector<ProcessInfo> expect{{1, "hz_app_aProcess", ProcessState::RUNNING},{0,"hz_app_bProcess", ProcessState::TERMINATED}};
		Check(instance, "Normal", expect);
	}

	{
		//修改DefaultMode为 Normal
		cout << "7.设置默认启动模式为Normal Mode" << endl;
		int32_t res = instance->SetDefaultMode("Normal");
		EXPECT_EQ(res, 0);

		cout << "1.SwitchMode to Driving" << endl;
		int32_t res3 = instance->SwitchMode("Driving");
		EXPECT_EQ(res3, 0);
		//判断
		vector<ProcessInfo> expect{{0, "hz_app_aProcess", ProcessState::TERMINATED},{1, "hz_app_bProcess", ProcessState::RUNNING}};
		Check(instance, "Driving", expect);
	}
	
}

int main(int argc, char* argv[])
{
	hozon::netaos::log::InitLogging("neta_dbg","neta_dbg",hozon::netaos::log::LogLevel::kInfo,
			hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE, "./", 10, 20);
	hozon::netaos::log::CreateLogger("-C", "", hozon::netaos::log::LogLevel::kOff);
	testing::InitGoogleTest(&argc,argv);
    return  RUN_ALL_TESTS();
}
