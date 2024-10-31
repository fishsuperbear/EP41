#include <thread>
#include <chrono>
#include <cstdint>
#include <signal.h>
#include <iostream>
#include <unistd.h>
#include "sm/include/state_client.h"
using namespace std;

using namespace hozon::netaos::sm;

class SMProcB {
public:
	SMProcB();
	~SMProcB();
	static int32_t preProcess_Normal_Driving(const std::string& old_mode, const std::string& new_mode);
	static void postProcess_Normal_Driving(const std::string& old_mode, const std::string& new_mode, const bool succ);
	static int32_t preProcess_Driving_Normal(const std::string& old_mode, const std::string& new_mode);
	static void postProcess_Driving_Normal(const std::string& old_mode, const std::string& new_mode, const bool succ);
	static int32_t preProcess_Normal_Parking(const std::string& old_mode, const std::string& new_mode);
	static void postProcess_Normal_Parking(const std::string& old_mode, const std::string& new_mode, const bool succ);
	static int32_t preProcess_Parking_Normal(const std::string& old_mode, const std::string& new_mode);
	static void postProcess_Parking_Normal(const std::string& old_mode, const std::string& new_mode, const bool succ);
	static int32_t preProcess_Normal_OTA(const std::string& old_mode, const std::string& new_mode);
	static void postProcess_Normal_OTA(const std::string& old_mode, const std::string& new_mode, const bool succ);
	static int32_t preProcess_OTA_Normal(const std::string& old_mode, const std::string& new_mode);
	static void postProcess_OTA_Normal(const std::string& old_mode, const std::string& new_mode, const bool succ);
	static int32_t preProcess_Driving_Parking(const std::string& old_mode, const std::string& new_mode);
	static void postProcess_Driving_Parking(const std::string& old_mode, const std::string& new_mode, const bool succ);
	static int32_t preProcess_Parking_Driving(const std::string& old_mode, const std::string& new_mode);
	static void postProcess_Parking_Driving(const std::string& old_mode, const std::string& new_mode, const bool succ);
	static int32_t preProcess_Driving_OTA(const std::string& old_mode, const std::string& new_mode);
	static void postProcess_Driving_OTA(const std::string& old_mode, const std::string& new_mode, const bool succ);
	static int32_t preProcess_OTA_Driving(const std::string& old_mode, const std::string& new_mode);
	static void postProcess_OTA_Driving(const std::string& old_mode, const std::string& new_mode, const bool succ);
	static int32_t preProcess_Parking_OTA(const std::string& old_mode, const std::string& new_mode);
	static void postProcess_Parking_OTA(const std::string& old_mode, const std::string& new_mode, const bool succ);
	static int32_t preProcess_OTA_Parking(const std::string& old_mode, const std::string& new_mode);
	static void postProcess_OTA_Parking(const std::string& old_mode, const std::string& new_mode, const bool succ);

	void doinit();
private:
	StateClient s_client;
};