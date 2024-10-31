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
测试场景：启动默认模式成功后，停止当前模式。
期望结果: EM成功退出模式下所有的进程。
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

TEST_F(FuncTest, StartAndStopNormalMode) {

	std::string _mod = "Normal";

    cout << "SwitchMode" << endl;
	int32_t res1 = m_instance->SwitchMode("Normal");
	EXPECT_EQ(res1, 0);

	std::string curr_mod;
	int32_t res2 = m_instance->GetCurrMode(curr_mod);
	cout << "GetCurrMode " << curr_mod << ",ret:" << res2 << endl;
	EXPECT_EQ(curr_mod, "Normal");

	int32_t res3 = m_instance->StopMode();
	cout << "stop mode,ret:"<< res3 << endl;
	sleep(10);
	EXPECT_EQ(res1, 0);

	std::string res="";
    char buf[128]={0};
    FILE * fp = popen("ps -ef | grep -v grep 'em_proc_' | awk '{print $2}' | wc -l", "r");
    if(fp){
        char* ret = fgets(buf,sizeof(buf),fp);
        if(ret && buf[0] != '\0' && buf[0] != '\n'){
            res = buf;
        }
        pclose(fp); fp = nullptr;
    }
	cout<< "proc num:" <<res <<endl;
	EXPECT_EQ(std::atoi(res.c_str()), 0);

}


int main(int argc, char* argv[])
{
    hozon::netaos::log::InitLogging("unit_test","utest",hozon::netaos::log::LogLevel::kOff,
			hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE, "./", 10, 20);
	hozon::netaos::log::CreateLogger("-T", "", hozon::netaos::log::LogLevel::kTrace);
	testing::InitGoogleTest(&argc,argv);
    return  RUN_ALL_TESTS();
}
