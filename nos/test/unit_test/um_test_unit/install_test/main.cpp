#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <signal.h>
#include <unistd.h>
#include <memory>

#include "update_manager/log/update_manager_logger.h"
#include "update_manager/agent/update_agent.h"
#include "update_nttask_soc_installer.h"
#include "update_cttask_interface_update.h"
#include "update_cttask_interface_activate.h"
#include "update_cttask_interface_wait_status.h"
#include "update_cttask_wait_sensor_update.h"
#include "update_tmtask_delay_timer.h"

#include "gtest/gtest.h"

using namespace hozon::netaos::update;
std::shared_ptr<hozon::netaos::log::Logger> logout = nullptr;

class InstallTest:public ::testing::Test{

protected:
	void SetUp() override {
		UM_DEBUG << "call SetUp";
	}
	void TearDown() override {
		UM_DEBUG << "call TearDown";
	} 
protected:

};

TEST_F(InstallTest, SocUpdateTest)
{
	SoC_t soc{};
	UpdateNTTaskSocInstaller* task = new UpdateNTTaskSocInstaller(nullptr, nullptr, soc, true);
    auto ret = UpdateAgent::Instance().post(task);
	ASSERT_NE(ret, eContinue);
}

int main(int argc, char* argv[])
{
    hozon::netaos::log::InitLogging("unit_test","utest",hozon::netaos::log::LogLevel::kTrace, hozon::netaos::log::HZ_LOG2FILE, "./", 10, 20);
	UpdateManagerLogger::GetInstance().CreateLogger("install_test");
	UM_DEBUG << "test um install_test start ...";

	testing::InitGoogleTest(&argc,argv);
    return  RUN_ALL_TESTS();
}
