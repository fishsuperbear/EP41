/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
*
* @File: get_time_and_timer_test.cpp
* @Date: 2023/12/08
* @Author: cheng
* @Desc: --
*/
//#define private public
//#define protected public
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <filesystem>

#include "log/include/logging.h"
#include "utils/include/path_utils.h"
#include "utils/include/time_utils.h"
#include "common/timer/timer_manager.hpp"
#include "gtest/gtest.h"

using namespace hozon::netaos::dc;

class GetTimerTest:public ::testing::Test {

   protected:
    static void SetUpTestSuite() {
        std::cout << "=== SetUpTestSuite ===" << std::endl;
    }

    static void TearDownTestSuite() {
        std::cout << "=== TearDownTestSuite ===" << std::endl;
    }

    void SetUp() override {
        //        system("export LD_LIBRARY_PATH=/app/lib:$LD_LIBRARY_PATH");
    }

    void TearDown() override {}

};


TEST_F(GetTimerTest, testcase_DATA_COLLECTION_ADD_DATA_001) {
     struct timespec ts {};
    TimerManager tm;
    ThreadPoolFlex tp(2,3);
    std::atomic_bool stop(false);
    std::atomic_int count{0};
    tm.start(tp,"testTimer");
    tm.addTimer(TaskPriority::LOW, 3000, 3, 1000, [&count] {
        count++;
    });
    for (int i=0;i<6;i++) {
        TimeUtils::sleep(498);
        EXPECT_EQ(count,0);
    }
    TimeUtils::sleep(62);  // 3050 s
    EXPECT_EQ(count,1);
    TimeUtils::sleep(1000);  // 4050 s
    EXPECT_EQ(count,2);
    TimeUtils::sleep(1000);  // 5050 s
    EXPECT_EQ(count,3);
    tp.stop();
    tm.stopAll();
}

int main(int argc, char* argv[]){
    hozon::netaos::log::InitLogging("DCTEST", "NETAOS DC", hozon::netaos::log::LogLevel::kTrace,
                                    hozon::netaos::log::HZ_LOG2CONSOLE , "/opt/usr/log/soc_log/", 10, (20*1024*1024),true);
    testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}
