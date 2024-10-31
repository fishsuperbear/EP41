#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <cstring>
#include <signal.h>
#include "em/include/proctypes.h"
#include "em/include/exec_client.h"
#include "em/include/emlogger.h"
#include "main.h"

std::string id;

using namespace hozon::netaos::em;

sig_atomic_t g_stopFlag = 0;


SMProcA::SMProcA() {

};
SMProcA::~SMProcA() {

}

int32_t SMProcA::preProcess(const std::string& old_mode, const std::string& new_mode) {
	cout << "do preProcess()!!!! id is " << id <<endl;
	postProcess(old_mode, new_mode, true);
	
	return 0;
}

void SMProcA::postProcess(const std::string& old_mode, const std::string& new_mode, const bool succ) {
	if (succ) {
		cout << "do postProcess()!!!! succ is true" << id <<endl;

	} else {
		cout << "do postProcess()!!!! succ is false" << id <<endl;

	}
}

void SMProcA::doinit() {

	std::string old_mod = "Normal";
	std::string new_mod = "Factory";
	cout << "RegisterPreProcessFunc" << endl;
	s_client.RegisterPreProcessFunc(old_mod, new_mod, &SMProcA::preProcess);
	cout << "RegisterPostProcessFunc" << endl;
	s_client.RegisterPostProcessFunc(old_mod, new_mod, &SMProcA::postProcess);
} 

int32_t SMProcA::switchmode() {
	cout << "SwitchMode to Factory" << endl;
	s_client.SwitchMode("Factory");
	return 0;
}

int32_t SMProcA::getcurrentmode() {
	std::string currmode;
	auto res = s_client.GetCurrMode(currmode);
	
	cout << "res is " << res << endl;
	cout << "currmode is " << currmode << endl;
	return 0;
}

int32_t SMProcA::setdefaultmode() {
	std::string defaultmode = "Normal";
	auto res = s_client.SetDefaultMode(defaultmode);
	
	cout << "res is " << res << endl;
	return 0;
}


void HandlerSignal(int32_t sig)
{
    std::cout << "proc a sig:<<"<< sig << std::endl;
    g_stopFlag = 1;
}

void ActThread()
{
    SMProcA sm_test_a;
	cout << "\n\n\n===============doinit==============" << endl;
	sm_test_a.doinit();
    sleep(10);
	cout << "\n\n\n===============getcurrentmode1==============" << endl;
    sm_test_a.getcurrentmode();
    sleep(10);
	cout << "\n\n\n===============switchmode==============" << endl;
	sm_test_a.switchmode();
    sleep(10);
	cout << "\n\n\n===============getcurrentmode2==============" << endl;
    sm_test_a.getcurrentmode();
    sleep(10);
	cout << "\n\n\n===============setdefaultmode==============" << endl;
    sm_test_a.setdefaultmode();
	cout << "===============over==============" << endl;

    while (!g_stopFlag) {
        std::this_thread::sleep_for(std::chrono::seconds(2u));
    }
}

void InitLog()
{
    EMLogger::GetInstance().InitLogging("proA","sm proc a",
        EMLogger::LogLevelType::LOG_LEVEL_OFF,
        hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE,
        "./", 10, 20
    );
    EMLogger::GetInstance().CreateLogger("-A");
}

int main(int argc, char* argv[])
{
    signal(SIGTERM, HandlerSignal);
    InitLog();

    std::shared_ptr<ExecClient> execli = std::make_shared<ExecClient>();
    int32_t ret = execli->ReportState(ExecutionState::kRunning);
    if(ret){ std::cout << "a report fail." << std::endl; }

    std::thread act(ActThread);
    act.join();

    ret = execli->ReportState(ExecutionState::kTerminating);
    return 0;
}