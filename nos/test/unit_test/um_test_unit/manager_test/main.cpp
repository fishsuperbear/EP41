#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <signal.h>
#include <unistd.h>
#include <memory>

#include "update_manager/log/update_manager_logger.h"
#include "update_manager/manager/uds_command_controller.h"

#include "gtest/gtest.h"

using namespace hozon::netaos::update;
std::shared_ptr<hozon::netaos::log::Logger> logout = nullptr;

class ManagerTest:public ::testing::Test{

protected:
	void SetUp() override {
		UM_DEBUG << "call SetUp";
	}
	void TearDown() override {
		UM_DEBUG << "call TearDown";
	} 
protected:

};

TEST_F(ManagerTest, commandControlTest)
{
	UdsCommandController::Instance()->Init();
	std::string cmd1 = "31 01 02 03";
	std::string cmd2 = "2E F1 98";
	std::string cmd3 = "2E F1 99";
	std::string cmd4 = "31 01 FF 01";
	std::string cmd5 = "31 01 02 05";

	auto res1 = UdsCommandController::Instance()->ProcessCommand(cmd1);
	auto res2 = UdsCommandController::Instance()->ProcessCommand(cmd2);
	auto res3 = UdsCommandController::Instance()->ProcessCommand(cmd3);
	auto res4 = UdsCommandController::Instance()->ProcessCommand(cmd4);
	auto res5 = UdsCommandController::Instance()->ProcessCommand(cmd5);

	ASSERT_EQ(res1, true);
	ASSERT_EQ(res2, true);
	ASSERT_EQ(res3, true);
	ASSERT_EQ(res4, true);
	ASSERT_EQ(res5, true);

	UdsCommandController::Instance()->Deinit();
}

int main(int argc, char* argv[])
{
    hozon::netaos::log::InitLogging("unit_test","utest",hozon::netaos::log::LogLevel::kTrace, hozon::netaos::log::HZ_LOG2FILE, "./", 10, 20);
	UpdateManagerLogger::GetInstance().CreateLogger("manager_test");
	UM_DEBUG << "test um manager_test start ...";

	testing::InitGoogleTest(&argc,argv);
    return  RUN_ALL_TESTS();
}
