#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <signal.h>
#include <unistd.h>
#include <memory>

#include "update_manager/log/update_manager_logger.h"
#include "update_manager/update_check/update_check.h"
#include "update_manager/config/update_settings.h"

#include "gtest/gtest.h"

using namespace hozon::netaos::update;
std::shared_ptr<hozon::netaos::log::Logger> logout = nullptr;

class CheckTest:public ::testing::Test{

protected:
	void SetUp() override {
		UM_DEBUG << "call SetUp";
	}
	void TearDown() override {
		UM_DEBUG << "call TearDown";
	} 
protected:

};

// 前置条件检查， 目前是检查空间
TEST_F(CheckTest, PreCheckTest)
{
	UpdateSettings::Instance().Init();
	UpdateCheck::Instance().Init();
	auto res = UpdateCheck::Instance().UpdatePreConditionCheck();
	ASSERT_EQ(res, -1);

	UpdateCheck::Instance().Deinit();
	UpdateSettings::Instance().Deinit();
}
// 获取默认的底盘信息
TEST_F(CheckTest, GetChassisInfoTest)
{
	UpdateCheck::Instance().Init();
	auto gear = UpdateCheck::Instance().GetGear();
	auto speed = UpdateCheck::Instance().GetSpeed();
	auto space = UpdateCheck::Instance().GetSpaceEnough();

	ASSERT_EQ(gear, 0);
	ASSERT_EQ(speed, -1);
	ASSERT_EQ(space, false);

	UpdateCheck::Instance().Deinit();
}
// 切换升级模式，不做详细测试，应该由SM模块测试进行包含
TEST_F(CheckTest, ChangeModeTest)
{
	UpdateCheck::Instance().Init();
	std::string mode{"Normal"};

	auto res = UpdateCheck::Instance().UpdateModeChange(mode);
	ASSERT_EQ(res, -1);
	
	UpdateCheck::Instance().Deinit();
}

int main(int argc, char* argv[])
{
    hozon::netaos::log::InitLogging("unit_test","utest",hozon::netaos::log::LogLevel::kTrace, hozon::netaos::log::HZ_LOG2FILE, "./", 10, 20);
	UpdateManagerLogger::GetInstance().CreateLogger("check_test");
	UM_DEBUG << "test um check_test start ...";

	testing::InitGoogleTest(&argc,argv);
    return  RUN_ALL_TESTS();
}
