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
测试场景：启停默认模式，进程延迟上报退出状态
期望结果: 超时上报Terminating的进程状态是Aborted
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


TEST_F(FuncTest, VerfiyProcStateByDelayReportTerminate) {
    std::string curr_mod;
    int32_t res2 = m_instance->GetCurrMode(curr_mod);
	EXPECT_EQ(res2, 0);
	cout << "GetCurrMode result is " << curr_mod << endl;
	EXPECT_EQ(curr_mod, "Normal");

    /* check proc running state */
	std::vector<ProcessInfo> vect;
	m_instance->GetProcessInfo(vect);
	bool isProcRunning = true;
	for (size_t i = 0; i < vect.size(); i++){
		ProcessInfo info = vect[i];
        if(info.procname.find(proc_vec[0]) != info.procname.npos && (uint32_t)info.procstate != 2){
            isProcRunning = false;
			cout << "start proc:" << info.procname << " is not running, cur state is "<<(uint32_t)info.procstate<< endl;
			break;
		}
	}
	EXPECT_EQ((int32_t)isProcRunning, 1);

    /* stop cur mode */
    m_instance->StopMode();
	sleep(EM_WAIT_TIME*2);

    /* check proc terminate state */
	vect.clear();
	m_instance->GetProcessInfo(vect);
	bool isProcAborted = true;
	for (size_t i = 0; i < vect.size(); i++){
		ProcessInfo info = vect[i];
        if(info.procname.find(proc_vec[0]) != info.procname.npos && (uint32_t)info.procstate != 5){
            isProcRunning = false;
			cout << "start proc:" << info.procname << " should be aborted, cur state is "<<(uint32_t)info.procstate<< endl;
			break;
		}
	}
    EXPECT_EQ((int32_t)isProcAborted, 1);
}


int main(int argc, char* argv[])
{
    hozon::netaos::log::InitLogging("unit_test","utest",hozon::netaos::log::LogLevel::kOff,
			hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE, "./", 10, 20);
	hozon::netaos::log::CreateLogger("-T", "", hozon::netaos::log::LogLevel::kTrace);
	testing::InitGoogleTest(&argc,argv);
    return  RUN_ALL_TESTS();
}
