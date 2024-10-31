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
测试场景：启动默认模式，进程延迟上报Running状态
期望结果: 超时上报Running的进程状态是Aborted
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
		SetConf();
        int res = system("chmod +x ../scripts/teardown.sh; ../scripts/teardown.sh");
		cout << res << endl;
		sleep(EM_WAIT_TIME);
    }

    void SetConf(std::string opt = ""){
		std::string cmd = "chmod +x ../scripts/setconf.sh; ../scripts/setconf.sh " + opt;
        int res = system(cmd.c_str());
		cout << res << endl;
	}

	bool searchProc(std::vector<std::string> &vect, std::string procname){
        std::vector<std::string>::iterator ite;
        ite = find(vect.begin(),vect.end(),procname);
        if(ite != vect.end()){
            return true;
        }
		return false;
	}
	void ResetConfig(std::string key){
		SetConf();
	    SetConf(key);
    }
protected:
	std::unique_ptr<StateClientImplZmq> m_instance;
	pid_t m_pid;
    vector<std::string> proc_vec = {"hz_app_bProcess"};
};


TEST_F(FuncTest, VerfiyProcStateByDelayReportRunning) {
	std::string curr_mod;
    int32_t res1 = m_instance->GetCurrMode(curr_mod);
	cout << "get curr mode:" << curr_mod << ",res:" << res1 << endl;
	EXPECT_EQ(curr_mod, "Normal");

    /* check proc running state */
	std::vector<ProcessInfo> vect;
	m_instance->GetProcessInfo(vect);
    size_t n = vect.size();
	bool isAppAborted = true;
	for (size_t i = 0; i < n; i++){
		ProcessInfo info = vect[i];
        if( info.procname.find(proc_vec[0]) != info.procname.npos && (uint32_t)info.procstate != 5){
            isAppAborted = false;
			cout << "start proc:" << info.procname << " should be aborted, cur state is "<<(uint32_t)info.procstate<< endl;
			break;
		}
	}
	EXPECT_EQ((int32_t)isAppAborted, 1);

}

int main(int argc, char* argv[])
{
    hozon::netaos::log::InitLogging("unit_test","utest",hozon::netaos::log::LogLevel::kOff,
			hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE, "./", 10, 20);
	hozon::netaos::log::CreateLogger("-T", "", hozon::netaos::log::LogLevel::kTrace);
	testing::InitGoogleTest(&argc,argv);
    return  RUN_ALL_TESTS();
}
