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
#include "em/include/emlogger.h"
#include "sm/include/state_client_zmq.h"


/***************测试说明***************/
/*
测试场景：同一个执行文件，配置不同启动参数和进程名。
期望结果: EM根据参数和进程名不同,运行多个进程。
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

TEST_F(FuncTest, VerfiyProcWithDiffParams) {
	std::string _mod = "Normal";

    cout << "SwitchMode" << endl;
	int32_t res1 = m_instance->SwitchMode("Normal");
	EXPECT_EQ(res1, 0);

	std::string curr_mod;
	m_instance->GetCurrMode(curr_mod);
	cout << "GetCurrMode result is " << curr_mod << endl;
	EXPECT_EQ(curr_mod, "Normal");

	std::string res="";
    char buf[128]={0};
    FILE * fp = popen("ps -ef | grep 'em_proc_a -m 200'| grep -v grep | awk '{print $2}' | wc -l", "r");
    if(fp){
        char* ret = fgets(buf,sizeof(buf),fp);
        if(ret && buf[0] != '\0' && buf[0] != '\n'){
            res = buf;
        }
        pclose(fp); fp = nullptr;
    }
	cout<< "proc num:" <<res <<endl;
	EXPECT_GE(std::atoi(res.c_str()), 1);
}

int main(int argc, char* argv[])
{
    hozon::netaos::log::InitLogging("unit_test","utest",hozon::netaos::log::LogLevel::kOff,
			hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE, "./", 10, 20);
	hozon::netaos::log::CreateLogger("-T", "", hozon::netaos::log::LogLevel::kTrace);
	testing::InitGoogleTest(&argc,argv);
    return  RUN_ALL_TESTS();
}
