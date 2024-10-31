#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <signal.h>
#include <unistd.h>
#include <memory>

#include "update_manager/log/update_manager_logger.h"
#include "update_manager/config/config_manager.h"
#include "update_manager/config/hz_xml.h"
#include "update_manager/config/update_settings.h"

#include "gtest/gtest.h"

using namespace hozon::netaos::update;
using namespace std;

class ConfigTest:public ::testing::Test{

protected:
	void SetUp() override {
		UM_DEBUG << "call SetUp";
	}
	void TearDown() override {
		UM_DEBUG << "call TearDown";
	}  
protected:
};

TEST_F(ConfigTest, parseSetting)
{
	UpdateSettings::Instance().Init();

	EXPECT_EQ(UpdateSettings::Instance().setting_file(), "/app/runtime_service/update_manager/conf/update_setting.json");
	UpdateSettings::Instance().update_zip_file("123");
	EXPECT_EQ(UpdateSettings::Instance().zip_file(), "123");
	EXPECT_EQ(UpdateSettings::Instance().path_for_unzip(), "/ota/hz_update/package_unzip/");
	EXPECT_EQ(UpdateSettings::Instance().path_for_bin_files(), "/ota/hz_update/bin_files/");
	EXPECT_EQ(UpdateSettings::Instance().path_for_update_tmp(), "/ota/hz_update/tmp/");
	EXPECT_EQ(UpdateSettings::Instance().path_for_recovery(), "/ota/recovery/");
	EXPECT_EQ(UpdateSettings::Instance().path_for_upgrade(), "/ota/");
	EXPECT_EQ(UpdateSettings::Instance().path_for_work(), "/ota/hz_update/");

	UpdateSettings::Instance().Deinit();
}

int main(int argc, char* argv[])
{
    hozon::netaos::log::InitLogging("unit_test","utest",hozon::netaos::log::LogLevel::kTrace, hozon::netaos::log::HZ_LOG2FILE, "./", 10, 20);
	UpdateManagerLogger::GetInstance().CreateLogger("config_test");
	UM_DEBUG << "test um config_test start ...";

	testing::InitGoogleTest(&argc,argv);
    return  RUN_ALL_TESTS();
}
