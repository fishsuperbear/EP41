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
#include "sm/include/state_client_impl.h"

/***************测试说明***************/
/*
测试场景：测试未启动execution_manager服务端，即也不存在StateServer的场景。
期望结果: 客户端各函数应该都失败。
*/
std::string id;

using namespace hozon::netaos::em;

#define EM_WAIT_TIME 2
class StateClientTest:public ::testing::Test{
protected:
	void SetUp() override {
		instance = make_unique<StateClientImpl>();
	}

	void TearDown() override {
	}
protected:
	std::unique_ptr<StateClientImpl> instance;
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

TEST(StateClient, createInstance) {
	std::unique_ptr<StateClientImpl> instance = make_unique<StateClientImpl>();
	EXPECT_NE(instance, nullptr);
	instance.reset();
	EXPECT_EQ(instance, nullptr);
}

TEST_F(StateClientTest, RegisterPreProcess) {

	std::string old_mod = "Normal";
	std::string new_mod = "Factory";
	cout << "RegisterPreProcessFunc" << endl;

	int32_t res = instance->RegisterPreProcessFunc(old_mod, new_mod, &StateClientTest::preProcess);
	EXPECT_EQ(res, -5);
	std::string process_name = instance->GetProcessName();
	std::string func_name = process_name + "_" +  old_mod + "_" + new_mod + "__preprocess";
	int map_size = instance->PreProcessFuncMap.size();
	EXPECT_EQ(map_size, 1);
	bool bRes= instance->PreProcessFuncMap.find(func_name) != instance->PreProcessFuncMap.end();
	EXPECT_TRUE(bRes);
	cout << "RegisterPreProcessFunc End" << endl;
}

TEST_F(StateClientTest, RegisterPostProcess) {

	std::string old_mod = "Normal";
	std::string new_mod = "Factory";
	cout << "RegisterPostProcessFunc" << endl;

	int32_t res = instance->RegisterPostProcessFunc(old_mod, new_mod, &StateClientTest::postProcess);
	EXPECT_EQ(res, -5);
	std::string process_name = instance->GetProcessName();
	std::string func_name = process_name + "_" +  old_mod + "_" + new_mod + "__postprocess";
	int map_size = instance->PostProcessFuncMap.size();
	EXPECT_EQ(map_size, 1);
	bool bRes= instance->PostProcessFuncMap.find(func_name) != instance->PostProcessFuncMap.end();
	EXPECT_TRUE(bRes);
	cout << "RegisterPostProcessFunc End" << endl;
}

TEST_F(StateClientTest, SwitchMode) {

	std::string new_mod = "Factory";

	int32_t res = instance->SwitchMode(new_mod);
	EXPECT_EQ(res, -5);
}

TEST_F(StateClientTest, GetCurrMode) {

	std::string curr_mod;

	int32_t res = instance->GetCurrMode(curr_mod);
	EXPECT_EQ(res, -5);
	EXPECT_EQ(curr_mod, "");
}

TEST_F(StateClientTest, SetDefaultMode) {

	std::string def_mod = "Normal";

	int32_t res = instance->SetDefaultMode(def_mod);
	EXPECT_EQ(res, -5);
}


TEST_F(StateClientTest, GetProcessInfo) {

	vector<ProcessInfo> process_info;
	int32_t res = instance->GetProcessInfo(process_info);
	EXPECT_EQ(res, -5);
	int process_info_size = process_info.size();
	EXPECT_EQ(process_info_size, 0);
}

int main(int argc, char* argv[])
{
	hozon::netaos::log::InitLogging("neta_dbg","neta_dbg",hozon::netaos::log::LogLevel::kTrace,
			hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE, "./", 10, 20);
	hozon::netaos::log::CreateLogger("-C", "", hozon::netaos::log::LogLevel::kTrace);
	testing::InitGoogleTest(&argc,argv);
    return  RUN_ALL_TESTS();
}
