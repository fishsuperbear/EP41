#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <signal.h>
#include <unistd.h>
#include <memory>

#include "update_manager/log/update_manager_logger.h"
#include "update_manager/file_to_bin/file_to_bin.h"
#include "update_manager/file_to_bin/hex_to_bin.h"
#include "update_manager/file_to_bin/s19_to_bin.h"

#include "gtest/gtest.h"

using namespace hozon::netaos::update;
using namespace std;

class File2BinTest:public ::testing::Test{

protected:
	void SetUp() override {
		UM_DEBUG << "call SetUp";
	}
	void TearDown() override {
		UM_DEBUG << "call TearDown";
	}  
protected:
};

TEST_F(File2BinTest, FileToBinTest)
{
	FileToBin instance{};
	std::string hexFilePath{""};
	std::string binFileOutput{""};
	auto res = instance.Transition(hexFilePath, binFileOutput);
	ASSERT_EQ(res, true);
}

int main(int argc, char* argv[])
{
    hozon::netaos::log::InitLogging("unit_test","utest",hozon::netaos::log::LogLevel::kTrace, hozon::netaos::log::HZ_LOG2FILE, "./", 10, 20);
	UpdateManagerLogger::GetInstance().CreateLogger("file2bin_test");
	UM_DEBUG << "test um file2bin_test start ...";

	testing::InitGoogleTest(&argc,argv);
    return  RUN_ALL_TESTS();
}
