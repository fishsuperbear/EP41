// #define private public
// #define protected public

#include <csignal>
#include <iostream>
#include <gtest/gtest.h>
#include "system_monitor/include/common/md5.h"

using namespace hozon::netaos::system_monitor;

class TestMd5 : public testing::Test
{
    protected:
        void SetUp() {}
        void TearDown() {}
};

TEST_F(TestMd5, MD5_1)
{
    MD5 md5();
}

TEST_F(TestMd5, MD5_2)
{
    MD5 md5(nullptr, 0);
}

TEST_F(TestMd5, MD5_3)
{
    MD5 md5("");
}

TEST_F(TestMd5, toString)
{
    std::ifstream in("/app/conf/system_monitor_config.json", std::ios::binary);
    MD5 md5(in);
    md5.toString();
}