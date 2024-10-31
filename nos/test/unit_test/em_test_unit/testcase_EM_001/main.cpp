#include <iostream>
#include <thread>
#include <chrono>
#include <cstring>
#include <signal.h>
#include <unistd.h>
#include <memory>
#include "gtest/gtest.h"
#include "em/include/proctypes.h"
#include "em/include/exec_client.h"
#include "em/include/logger.h"
#include "sm/include/state_client_zmq.h"


/***************测试说明***************/
/*
测试场景：测试EM启动默认模式下进程组。
期望结果: 客户端获取启动模式成功（进程启动正常）
*/


using namespace hozon::netaos::em;
using namespace hozon::netaos::sm;

#define EM_WAIT_TIME 5

class FuncTest:public ::testing::Test{

protected:
	static void SetUpTestSuite(){
        cout << "=== SetUpTestSuite ===" << endl;
    }
	static void TearDownTestSuite(){
        cout << "=== TearDownTestSuite ===" << endl;
    }
	void SetUp() override{
        int res = system("chmod +x ../scripts/setup.sh; ../scripts/setup.sh");
		cout << res << endl;
		sleep(EM_WAIT_TIME);
        m_instance = make_unique<StateClientImplZmq>();
    }
	void TearDown() override{
        int res = system("chmod +x ../scripts/teardown.sh; ../scripts/teardown.sh");
		cout << res << endl;
		sleep(EM_WAIT_TIME);
    }  
protected:
	std::unique_ptr<StateClientImplZmq> m_instance;
	pid_t m_pid;

};

TEST_F(FuncTest, StartNormalMode) {

	std::string _mod = "Normal";

    cout << "SwitchMode" << endl;
	int32_t res1 = m_instance->SwitchMode("Normal");
	EXPECT_EQ(res1, 0);

	std::string curr_mod;
	int32_t res2 = m_instance->GetCurrMode(curr_mod);
	EXPECT_EQ(res2, 0);
	cout << "GetCurrMode result is " << curr_mod << endl;
	EXPECT_EQ(curr_mod, "Normal");
}


int main(int argc, char* argv[])
{
    hozon::netaos::log::InitLogging("unit_test","utest",hozon::netaos::log::LogLevel::kOff,
			hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE, "./", 10, 20);
	hozon::netaos::log::CreateLogger("-T", "", hozon::netaos::log::LogLevel::kTrace);
	testing::InitGoogleTest(&argc,argv);
    return  RUN_ALL_TESTS();
}
