#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <signal.h>
#include <unistd.h>
#include <memory>

#include "update_manager/log/update_manager_logger.h"
#include "update_manager/record/ota_record.h"
#include "update_manager/record/ota_store.h"

#include "gtest/gtest.h"

using namespace hozon::netaos::update;
std::shared_ptr<hozon::netaos::log::Logger> logout = nullptr;

class RecordTest:public ::testing::Test{

protected:
	void SetUp() override {
		UM_DEBUG << "call SetUp";
	}
	void TearDown() override {
		UM_DEBUG << "call TearDown";
	} 
protected:

};

TEST_F(RecordTest, RecordTest01)
{
	OTARecoder::Instance().Init();
	// TODO
	OTARecoder::Instance().Deinit();
}


int main(int argc, char* argv[])
{
    hozon::netaos::log::InitLogging("unit_test","utest",hozon::netaos::log::LogLevel::kTrace, hozon::netaos::log::HZ_LOG2FILE, "./", 10, 20);
	UpdateManagerLogger::GetInstance().CreateLogger("record_test");
	UM_DEBUG << "test um record_test start ...";

	testing::InitGoogleTest(&argc,argv);
    return  RUN_ALL_TESTS();
}
