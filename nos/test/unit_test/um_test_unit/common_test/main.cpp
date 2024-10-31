#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <signal.h>
#include <unistd.h>
#include <memory>

#include "update_manager/log/update_manager_logger.h"
#include "update_manager/common/common_operation.h"

#include "gtest/gtest.h"

using namespace hozon::netaos::update;
std::shared_ptr<hozon::netaos::log::Logger> logout = nullptr;

class CommonTest:public ::testing::Test{

protected:
	void SetUp() override {
		UM_DEBUG << "call SetUp";
	}
	void TearDown() override {
		UM_DEBUG << "call TearDown";
	} 
protected:

};

TEST_F(CommonTest, PathExists)
{
	std::string file = "/home/xiaoyu/work/netaos/nos/test/unit_test/um_test_unit/common_test/main.cpp";
	auto res = PathExists(file);
	ASSERT_EQ(res, true);
}

TEST_F(CommonTest, PathNotExists)
{
	std::string file = "/home/xiaoyu/work/netaos/nos/test/unit_test/um_test_unit/common_test/main.txt";
	auto res = PathExists(file);
	ASSERT_EQ(res, false);
}


int main(int argc, char* argv[])
{
    hozon::netaos::log::InitLogging("unit_test","utest",hozon::netaos::log::LogLevel::kTrace, hozon::netaos::log::HZ_LOG2FILE, "./", 10, 20);
	UpdateManagerLogger::GetInstance().CreateLogger("common_test");
	UM_DEBUG << "test um common_test start ...";

	testing::InitGoogleTest(&argc,argv);
    return  RUN_ALL_TESTS();
}
