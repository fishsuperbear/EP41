#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <signal.h>
#include <unistd.h>
#include <memory>

#include "update_manager/log/update_manager_logger.h"
#include "update_manager/state_machine/state_file_manager.h"
#include "update_manager/state_machine/update_state_machine.h"
#include "update_manager/config/update_settings.h"

#include "gtest/gtest.h"

using namespace hozon::netaos::update;
std::shared_ptr<hozon::netaos::log::Logger> logout = nullptr;

class StateTest:public ::testing::Test{

protected:
	void SetUp() override {
		UM_DEBUG << "call SetUp";
	}
	void TearDown() override {
		UM_DEBUG << "call TearDown";
	} 
protected:

};

TEST_F(StateTest, CreateStateFileTest)
{
	UpdateSettings::Instance().Init();

	StateFileManager::Instance()->Init();
	auto res = StateFileManager::Instance()->CreateStateFile();

	ASSERT_EQ(res, true);
	
	StateFileManager::Instance()->Deinit();
	UpdateSettings::Instance().Deinit();
}

int main(int argc, char* argv[])
{
    hozon::netaos::log::InitLogging("unit_test","utest",hozon::netaos::log::LogLevel::kTrace, hozon::netaos::log::HZ_LOG2FILE, "./", 10, 20);
	UpdateManagerLogger::GetInstance().CreateLogger("state_test");
	UM_DEBUG << "test um state_test start ...";

	testing::InitGoogleTest(&argc,argv);
    return  RUN_ALL_TESTS();
}
