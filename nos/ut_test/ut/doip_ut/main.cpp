
#include <iostream>
#include "gtest/gtest.h"
#include "log/include/logging.h"
#include "base/doip_logger.h"

int main(int argc, char* argv[])
{
    hozon::netaos::log::InitLogging("gtest_doip","gtest_doip",hozon::netaos::log::LogLevel::kTrace,
            hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE, "./", 10, 20);
    auto log = hozon::netaos::log::CreateLogger("gtest_doip", "", hozon::netaos::log::LogLevel::kTrace);
    log->LogInfo() << "start................";

    //hozon::netaos::diag::DoIPLogger::GetInstance().CreateLogger("doip");

    testing::InitGoogleTest(&argc,argv);
    int ret = RUN_ALL_TESTS();
    std::cout << "====RUN_ALL_TESTS ret " << ret << std::endl;
    
    return 0;
}


