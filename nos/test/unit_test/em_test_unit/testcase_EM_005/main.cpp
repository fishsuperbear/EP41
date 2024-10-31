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
测试场景：启停默认模式后切换模式。
期望结果: 启动模式成功后进程状态为Running，切换模式后，退出的进程和新启动的进程符合预期
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
	bool searchProc(std::vector<std::string> &vect, std::string procname){
        std::vector<std::string>::iterator ite;
        ite = find(vect.begin(),vect.end(),procname);
        if(ite != vect.end()){
            return true;
        }
		return false;
	}
protected:
	std::unique_ptr<StateClientImplZmq> m_instance;
	pid_t m_pid;

};


TEST_F(FuncTest, StartAndVerifySwitchMode) {

	std::string _mod = "Normal";

	vector<std::string> proc_vec = {"hz_app_eProcess","hz_app_jProcess", "hz_app_oProcess"};

    cout << "SwitchMode" << endl;
	int32_t res1 = m_instance->SwitchMode("Normal");
	EXPECT_EQ(res1, 0);

	std::string curr_mod;
	int32_t res2 = m_instance->GetCurrMode(curr_mod);
	// EXPECT_EQ(res2, 0);
	cout << "GetCurrMode result is " << curr_mod <<",ret:" <<res2 << endl;
	EXPECT_EQ(curr_mod, "Normal");

    /* check proc running state */
	std::vector<ProcessInfo> vect;
	m_instance->GetProcessInfo(vect);
    size_t n = vect.size();
	bool isAllRunning = true;
	for (size_t i = 0; i < n; i++){
		ProcessInfo info = vect[i];
        if( info.procname.find("hz_app") != info.procname.npos && (uint32_t)info.procstate != 2){
            isAllRunning = false;
			cout << "proc:" << info.procname << " is not running, cur state is "<<(uint32_t)info.procstate<< endl;
			break;
		}
	}
	EXPECT_EQ((int32_t)isAllRunning, 1);

    /* switch mode parking */
	res1 = m_instance->SwitchMode("Parking");
	EXPECT_EQ(res1, 0);

    curr_mod = "";
    res2 = m_instance->GetCurrMode(curr_mod);
	// EXPECT_EQ(res2, 0);
	cout << "GetCurrMode result is " << curr_mod <<",ret:" <<res2 << endl;
	EXPECT_EQ(curr_mod, "Parking");

    /* check proc running state */
	m_instance->GetProcessInfo(vect);
    size_t n1 = vect.size();
	isAllRunning = true;
	bool isProcFTerminated = true;
	for (size_t i = 0; i < n1; i++){
		ProcessInfo info = vect[i];
        if(info.procname.find("hz_app") != info.procname.npos && !searchProc(proc_vec,info.procname) && (uint32_t)info.procstate != 2){
            isAllRunning = false;
			cout << "proc:" << info.procname << " is not running, cur state is "<<(uint32_t)info.procstate<< endl;
			break;
		}
		else if(searchProc(proc_vec,info.procname) && (uint32_t)info.procstate != 4){
			cout << "proc:" << info.procname << " is not terminated, cur state is "<<(uint32_t)info.procstate<< endl;
			isProcFTerminated = false;
			break;
		}
	}
	EXPECT_EQ((int32_t)isAllRunning, 1);
    EXPECT_EQ((int32_t)isProcFTerminated, 1);
}


int main(int argc, char* argv[])
{
    hozon::netaos::log::InitLogging("unit_test","utest",hozon::netaos::log::LogLevel::kOff,
			hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE, "./", 10, 20);
	hozon::netaos::log::CreateLogger("-T", "", hozon::netaos::log::LogLevel::kTrace);
	testing::InitGoogleTest(&argc,argv);
    return  RUN_ALL_TESTS();
}
