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


SMProcB::SMProcB() {

};
SMProcB::~SMProcB() {

}

int32_t SMProcB::preProcess_Normal_Driving(const std::string& old_mode, const std::string& new_mode) {
	_LOG_INFO << "SMProcB do preProcess_Normal_Driving()!!!! id is " << id;
	return 0;
}

void SMProcB::postProcess_Normal_Driving(const std::string& old_mode, const std::string& new_mode, const bool succ) {
	if (succ) {
		_LOG_INFO << "SMProcB do postProcess_Normal_Driving()!!!! succ is true" << id;
	} else {
		_LOG_INFO << "SMProcB do postProcess_Normal_Driving()!!!! succ is false" << id;
	}
}

int32_t SMProcB::preProcess_Driving_Normal(const std::string& old_mode, const std::string& new_mode) {
	_LOG_INFO << "SMProcB do preProcess_Driving_Normal()!!!! id is " << id;
	return 0;
}

void SMProcB::postProcess_Driving_Normal(const std::string& old_mode, const std::string& new_mode, const bool succ) {
	if (succ) {
		_LOG_INFO << "SMProcB do postProcess_Driving_Normal()!!!! succ is true" << id;
	} else {
		_LOG_INFO << "SMProcB do postProcess_Driving_Normal()!!!! succ is false" << id;
	}
}

int32_t SMProcB::preProcess_Normal_Parking(const std::string& old_mode, const std::string& new_mode) {
	_LOG_INFO << "SMProcB do preProcess_Normal_Parking()!!!! id is " << id;
	return 0;
}

void SMProcB::postProcess_Normal_Parking(const std::string& old_mode, const std::string& new_mode, const bool succ) {
	if (succ) {
		_LOG_INFO << "SMProcB do postProcess_Normal_Parking()!!!! succ is true" << id;
	} else {
		_LOG_INFO << "SMProcB do postProcess_Normal_Parking()!!!! succ is false" << id;
	}
}

int32_t SMProcB::preProcess_Parking_Normal(const std::string& old_mode, const std::string& new_mode) {
	_LOG_INFO << "SMProcB do preProcess_Parking_Normal()!!!! id is " << id;
	return 0;
}

void SMProcB::postProcess_Parking_Normal(const std::string& old_mode, const std::string& new_mode, const bool succ) {
	if (succ) {
		_LOG_INFO << "SMProcB do postProcess_Parking_Normal()!!!! succ is true" << id;
	} else {
		_LOG_INFO << "SMProcB do postProcess_Parking_Normal()!!!! succ is false" << id;
	}
}

int32_t SMProcB::preProcess_Normal_OTA(const std::string& old_mode, const std::string& new_mode) {
	_LOG_INFO << "SMProcB do preProcess_Normal_OTA()!!!! id is " << id;
	return 0;
}

void SMProcB::postProcess_Normal_OTA(const std::string& old_mode, const std::string& new_mode, const bool succ) {
	if (succ) {
		_LOG_INFO << "SMProcB do postProcess_Normal_OTA()!!!! succ is true" << id;
	} else {
		_LOG_INFO << "SMProcB do postProcess_Normal_OTA()!!!! succ is false" << id;
	}
}

int32_t SMProcB::preProcess_OTA_Normal(const std::string& old_mode, const std::string& new_mode) {
	_LOG_INFO << "SMProcB do preProcess_OTA_Normal()!!!! id is " << id;
	return 0;
}

void SMProcB::postProcess_OTA_Normal(const std::string& old_mode, const std::string& new_mode, const bool succ) {
	if (succ) {
		_LOG_INFO << "SMProcB do postProcess_OTA_Normal()!!!! succ is true" << id;
	} else {
		_LOG_INFO << "SMProcB do postProcess_OTA_Normal()!!!! succ is false" << id;
	}
}

int32_t SMProcB::preProcess_Driving_Parking(const std::string& old_mode, const std::string& new_mode) {
	_LOG_INFO << "SMProcB do preProcess_Driving_Parking()!!!! id is " << id;
	return 0;
}

void SMProcB::postProcess_Driving_Parking(const std::string& old_mode, const std::string& new_mode, const bool succ) {
	if (succ) {
		_LOG_INFO << "SMProcB do postProcess_Driving_Parking()!!!! succ is true" << id;
	} else {
		_LOG_INFO << "SMProcB do postProcess_Driving_Parking()!!!! succ is false" << id;
	}
}

int32_t SMProcB::preProcess_Parking_Driving(const std::string& old_mode, const std::string& new_mode) {
	_LOG_INFO << "SMProcB do preProcess_Parking_Driving()!!!! id is " << id;
	return 0;
}

void SMProcB::postProcess_Parking_Driving(const std::string& old_mode, const std::string& new_mode, const bool succ) {
	if (succ) {
		_LOG_INFO << "SMProcB do postProcess_Parking_Driving()!!!! succ is true" << id;
	} else {
		_LOG_INFO << "SMProcB do postProcess_Parking_Driving()!!!! succ is false" << id;
	}
}


int32_t SMProcB::preProcess_Driving_OTA(const std::string& old_mode, const std::string& new_mode) {
	_LOG_INFO << "SMProcB do preProcess_Driving_OTA()!!!! id is " << id;
	return 0;
}

void SMProcB::postProcess_Driving_OTA(const std::string& old_mode, const std::string& new_mode, const bool succ) {
	if (succ) {
		_LOG_INFO << "SMProcB do postProcess_Driving_OTA()!!!! succ is true" << id;
	} else {
		_LOG_INFO << "SMProcB do postProcess_Driving_OTA()!!!! succ is false" << id;
	}
}

int32_t SMProcB::preProcess_OTA_Driving(const std::string& old_mode, const std::string& new_mode) {
	_LOG_INFO << "SMProcB do preProcess_OTA_Driving()!!!! id is " << id;
	return 0;
}

void SMProcB::postProcess_OTA_Driving(const std::string& old_mode, const std::string& new_mode, const bool succ) {
	if (succ) {
		_LOG_INFO << "SMProcB do postProcess_OTA_Driving()!!!! succ is true" << id;
	} else {
		_LOG_INFO << "SMProcB do postProcess_OTA_Driving()!!!! succ is false" << id;
	}
}

int32_t SMProcB::preProcess_Parking_OTA(const std::string& old_mode, const std::string& new_mode) {
	_LOG_INFO << "SMProcB do preProcess_Parking_OTA()!!!! id is " << id;
	return 0;
}

void SMProcB::postProcess_Parking_OTA(const std::string& old_mode, const std::string& new_mode, const bool succ) {
	if (succ) {
		_LOG_INFO << "SMProcB do postProcess_Parking_OTA()!!!! succ is true" << id;
	} else {
		_LOG_INFO << "SMProcB do postProcess_Parking_OTA()!!!! succ is false" << id;
	}
}

int32_t SMProcB::preProcess_OTA_Parking(const std::string& old_mode, const std::string& new_mode) {
	_LOG_INFO << "SMProcB do preProcess_OTA_Parking()!!!! id is " << id;
	return 0;
}

void SMProcB::postProcess_OTA_Parking(const std::string& old_mode, const std::string& new_mode, const bool succ) {
	if (succ) {
		_LOG_INFO << "SMProcB do postProcess_OTA_Parking()!!!! succ is true" << id;
	} else {
		_LOG_INFO << "SMProcB do postProcess_OTA_Parking()!!!! succ is false" << id;
	}
}

void SMProcB::doinit() {

	_LOG_INFO << "SMProcB RegisterPreProcessFunc(Normal  ===>  Driving)";
	s_client.RegisterPreProcessFunc("Normal", "Driving", &SMProcB::preProcess_Normal_Driving);
	_LOG_INFO << "SMProcB RegisterPostProcessFunc(Normal  ===>  Driving)";
	s_client.RegisterPostProcessFunc("Normal", "Driving", &SMProcB::postProcess_Normal_Driving);
	_LOG_INFO << "SMProcB RegisterPreProcessFunc(Normal  ===>  Parking)";
	s_client.RegisterPreProcessFunc("Normal", "Parking", &SMProcB::preProcess_Normal_Parking);
	_LOG_INFO << "SMProcB RegisterPostProcessFunc(Normal  ===>  Parking)";
	s_client.RegisterPostProcessFunc("Normal", "Parking", &SMProcB::postProcess_Normal_Parking);
	_LOG_INFO << "SMProcB RegisterPreProcessFunc(Normal  ===>  OTA)";
	s_client.RegisterPreProcessFunc("Normal", "OTA", &SMProcB::preProcess_Normal_OTA);
	_LOG_INFO << "SMProcB RegisterPostProcessFunc(Normal  ===>  OTA)";
	s_client.RegisterPostProcessFunc("Normal", "OTA", &SMProcB::postProcess_Normal_OTA);
	_LOG_INFO << "SMProcB RegisterPreProcessFunc(Driving  ===>  Parking)";
	s_client.RegisterPreProcessFunc("Driving", "Parking", &SMProcB::preProcess_Driving_Parking);
	_LOG_INFO << "SMProcB RegisterPostProcessFunc(Driving  ===>  Parking)";
	s_client.RegisterPostProcessFunc("Driving", "Parking", &SMProcB::postProcess_Driving_Parking);
	_LOG_INFO << "SMProcB RegisterPreProcessFunc(Driving  ===>  OTA)";
	s_client.RegisterPreProcessFunc("Driving", "OTA", &SMProcB::preProcess_Driving_OTA);
	_LOG_INFO << "SMProcB RegisterPostProcessFunc(Driving  ===>  OTA)";
	s_client.RegisterPostProcessFunc("Driving", "OTA", &SMProcB::postProcess_Driving_OTA);
	_LOG_INFO << "SMProcB RegisterPreProcessFunc(Driving  ===>  Normal)";
	s_client.RegisterPreProcessFunc("Driving", "Normal", &SMProcB::preProcess_Driving_Normal);
	_LOG_INFO << "SMProcB RegisterPostProcessFunc(Driving  ===>  Normal)";
	s_client.RegisterPostProcessFunc("Driving", "Normal", &SMProcB::postProcess_Driving_Normal);
} 

void HandlerSignal(int32_t sig)
{
    std::cout << "proc a sig:<<"<< sig << std::endl;
    g_stopFlag = 1;
}

void ActThread()
{
    while (!g_stopFlag) {
        std::this_thread::sleep_for(std::chrono::seconds(2u));
    }
}

void InitLog()
{
    EMLogger::GetInstance().InitLogging("proB","sm proc b",
        EMLogger::LogLevelType::LOG_LEVEL_INFO,
        hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE,
        "/log/", 10, 20
    );
    EMLogger::GetInstance().CreateLogger("BB");
}

int main(int argc, char* argv[])
{
    signal(SIGTERM, HandlerSignal);
    InitLog();

    std::shared_ptr<ExecClient> execli(new ExecClient());
    int32_t ret = execli->ReportState(ExecutionState::kRunning);
    if(ret){ std::cout << "b report fail." << std::endl; }

	id = "---------------------------------";
	SMProcB sm_test_b;
	_LOG_INFO << "===============doinit==============";
	sm_test_b.doinit();

    std::thread act(ActThread);
    act.join();

    ret = execli->ReportState(ExecutionState::kTerminating);
    return 0;
}