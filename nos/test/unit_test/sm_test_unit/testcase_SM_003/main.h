#include <thread>
#include <chrono>
#include <cstdint>
#include <signal.h>
#include <iostream>
#include <unistd.h>
#include "sm/include/state_client.h"
using namespace std;

using namespace hozon::netaos::sm;

class SMProcA {
public:
	SMProcA();
	~SMProcA();
	static int32_t preProcess(const std::string& old_mode, const std::string& new_mode);
	static void postProcess(const std::string& old_mode, const std::string& new_mode, const bool succ);
	void doinit();
	int32_t switchmode();
	
	int32_t setdefaultmode();
	int32_t getcurrentmode();
private:
	StateClient s_client;
};