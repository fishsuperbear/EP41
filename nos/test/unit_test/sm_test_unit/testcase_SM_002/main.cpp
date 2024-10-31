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
测试场景：测试存在EM服务端，存在StateServer的场景。EM拉起了em_proc_a进程，客户端正向测试
期望结果: 客户端请求函数都成功。
*/
std::string id;

using namespace hozon::netaos::em;

#define EM_WAIT_TIME 6
class StateClientTest_HasServer:public ::testing::Test{
protected:
	static void SetUpTestSuite() {
		cout << "=========SetUpTestSuite=========" << endl;
	}
	static void TearDownTestSuite() {
		cout << "=========TearDownTestSuite=========" << endl;
	}
	void SetUp() override {
		//将需要由em拉起的进程拷贝到sm_test/emproc目录中
		int res = system("chmod +x ../scripts/SetUp.sh; ../scripts/SetUp.sh");
		cout << res << endl;
		sleep(EM_WAIT_TIME);
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

TEST_F(StateClientTest_HasServer, RegisterPreProcess_HasServer) {

	std::string old_mod = "Normal";
	std::string new_mod = "Factory";
	cout << "RegisterPreProcessFunc" << endl;

	int32_t res = instance->RegisterPreProcessFunc(old_mod, new_mod, &StateClientTest_HasServer::preProcess);
	sleep(2);
	EXPECT_EQ(res, 0);
	std::string process_name = instance->GetProcessName();
	std::string func_name = process_name + "_" +  old_mod + "_" + new_mod + "__preprocess";
	int map_size = instance->PreProcessFuncMap.size();
	EXPECT_EQ(map_size, 1);
	bool bRes= instance->PreProcessFuncMap.find(func_name) != instance->PreProcessFuncMap.end();
	EXPECT_TRUE(bRes);
	cout << "RegisterPreProcessFunc End" << endl;
}

TEST_F(StateClientTest_HasServer, RegisterPostProcess_HasServer) {

	std::string old_mod = "Normal";
	std::string new_mod = "Factory";
	cout << "RegisterPostProcessFunc" << endl;

	int32_t res = instance->RegisterPostProcessFunc(old_mod, new_mod, &StateClientTest_HasServer::postProcess);
	sleep(2);
	EXPECT_EQ(res, 0);
	std::string process_name = instance->GetProcessName();
	std::string func_name = process_name + "_" +  old_mod + "_" + new_mod + "__postprocess";
	int map_size = instance->PostProcessFuncMap.size();
	EXPECT_EQ(map_size, 1);
	bool bRes= instance->PostProcessFuncMap.find(func_name) != instance->PostProcessFuncMap.end();
	EXPECT_TRUE(bRes);
	cout << "RegisterPostProcessFunc End" << endl;
}

TEST_F(StateClientTest_HasServer, SwitchMode_HasServer) {

	std::string old_mod = "Normal";
	std::string new_mod = "Driving";
	cout << "RegisterPreProcessFunc" << endl;
	int32_t res1 = instance->RegisterPreProcessFunc(old_mod, new_mod, &StateClientTest_HasServer::preProcess);
	EXPECT_EQ(res1, 0);
	cout << "RegisterPostProcessFunc" << endl;
	int32_t res2 = instance->RegisterPostProcessFunc(old_mod, new_mod, &StateClientTest_HasServer::postProcess);
	EXPECT_EQ(res2, 0);
	sleep(2);
	
	cout << "SwitchMode" << endl;
	int32_t res3 = instance->SwitchMode(new_mod);
	EXPECT_EQ(res3, 0);

	std::string curr_mod;
	int32_t res = instance->GetCurrMode(curr_mod);
	EXPECT_EQ(res, 0);
	cout << "GetCurrMode result is " << curr_mod << endl;
	EXPECT_EQ(curr_mod, "Driving");
}

TEST_F(StateClientTest_HasServer, GetCurrMode_HasServer) {
	cout << "GetCurrMode" << endl;
	std::string curr_mod;

	int32_t res = instance->GetCurrMode(curr_mod);
	EXPECT_EQ(res, 0);
	cout << "GetCurrMode result is " << curr_mod << endl;
	EXPECT_EQ(curr_mod, "Normal");
}

TEST_F(StateClientTest_HasServer, SetDefaultMode_HasServer) {

	std::string def_mod = "Normal";
	int32_t res = instance->SetDefaultMode(def_mod);
	EXPECT_EQ(res, 0);
}

TEST_F(StateClientTest_HasServer, GetProcessInfo_HasServer) {

	vector<ProcessInfo> process_info;
	int32_t res = instance->GetProcessInfo(process_info);
	EXPECT_EQ(res, 0);
	int process_info_size = process_info.size();
	EXPECT_EQ(process_info_size, 1);
}

int main(int argc, char* argv[])
{
	hozon::netaos::log::InitLogging("neta_dbg","neta_dbg",hozon::netaos::log::LogLevel::kTrace,
			hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE, "./", 10, 20);
	hozon::netaos::log::CreateLogger("-C", "", hozon::netaos::log::LogLevel::kTrace);
	testing::InitGoogleTest(&argc,argv);
    return  RUN_ALL_TESTS();
}
